"""Nối các stage. Đọc file này để thấy toàn bộ luồng trên một trang."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .config import Config
from .kb import KnowledgeBase
from .llm import cache as llm_cache
from .llm.base import get_agent
from .schema import (AnalysisResult, CulturalSpec, EvalRecord, MultiGenResult, Prompt, ReviewOutcome, RunSummary,
                     SearchResult, from_dict, to_dict)
from .stages import analysis as st_analysis
from .stages import evaluation as st_eval
from .stages import extraction as st_extract
from .stages import review as st_review
from .stages import spec as st_spec
from .stages.generation import get_generator
from .stages.perception import get_perceiver
from .stages.retrieval import get_retriever


def load_prompts(path: str | Path) -> list[Prompt]:
    """Đọc .jsonl (một prompt một dòng) hoặc .json (danh sách, vd bộ complex của nhóm: id, text_vi, text_en, difficulty,
    entities -> gold_entities, categories_used -> category)."""
    text = Path(path).read_text(encoding="utf-8")
    if str(path).endswith(".json"):
        rows = json.loads(text)
        rows = rows if isinstance(rows, list) else rows.get("prompts", [])
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip() and not line.startswith("//")]
    out = []
    for r in rows:
        r = dict(r)
        ents = r.pop("entities", None)
        cats = r.pop("categories_used", None) or r.pop("categories", None)  # bộ 2026-09-15 của nhóm dùng 'categories'
        r.pop("categories", None)
        if ents and not r.get("gold_entities"):
            r["gold_entities"] = list(ents)
        if cats and not r.get("category"):
            r["category"] = ", ".join(cats) if isinstance(cats, list) else str(cats)
        if r.get("difficulty") not in ("easy", "medium", "hard"):
            r["difficulty"] = "medium"
        known = {"id", "text_vi", "text_en", "category", "difficulty", "gold_entities", "note"}
        out.append(Prompt(**{k: v for k, v in r.items() if k in known}))
    return out


@dataclass
class PipelineResult:
    prompt: Prompt
    analysis: AnalysisResult
    search: SearchResult
    spec: CulturalSpec
    outcome: ReviewOutcome
    record: EvalRecord
    multigen: MultiGenResult | None = None


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_dict(obj), ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- cache stage 1-3 (hàm module để notebook/session dùng chung)
def stage_cache_key(cfg: Config, prompt: Prompt, kb_version: str) -> str:
    c = cfg
    sig = json.dumps({
        "text": prompt.text_vi, "en": prompt.text_en,
        "llm": [c.llm.backend, c.llm.model],
        "retrieval": [c.retrieval.backend, c.retrieval.extract, c.retrieval.web_api, c.retrieval.wiki_chars],
        "spec": [c.max_spec_entities, c.min_entity_score, c.max_candidate_entities], "kb": kb_version,
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(sig.encode()).hexdigest()[:16]


def load_stage_cache(cache_dir: Path, key: str):
    d = cache_dir / "stages" / key
    try:
        a = from_dict(AnalysisResult, json.loads((d / "analysis.json").read_text(encoding="utf-8")))
        s = from_dict(SearchResult, json.loads((d / "search.json").read_text(encoding="utf-8")))
        sp = from_dict(CulturalSpec, json.loads((d / "spec.json").read_text(encoding="utf-8")))
        return a, s, sp
    except (OSError, json.JSONDecodeError, TypeError, KeyError):
        return None


def save_stage_cache(cache_dir: Path, key: str, prompt: Prompt, analysis, search, spec) -> None:
    d = cache_dir / "stages" / key
    _write(d / "analysis.json", analysis)
    _write(d / "search.json", search)
    _write(d / "spec.json", spec)
    (d / "prompt.txt").write_text(prompt.text_vi, encoding="utf-8")


class Pipeline:
    def __init__(self, cfg: Config, log=print):
        # Giảm phân mảnh VRAM (gợi ý trong chính thông báo OOM của PyTorch). Phải đặt trước khi CUDA khởi tạo.
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        self.cfg, self.log = cfg, log
        self.kb = KnowledgeBase.load(cfg.kb_path)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = cfg.run_name or f"{stamp}-{cfg.llm.backend}-{cfg.t2i.backend}"
        self.run_dir = Path(cfg.runs_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        _write(self.run_dir / "config.json", cfg.as_dict())
        self.cache_dir = Path(cfg.cache.dir) if cfg.cache.dir else Path(cfg.runs_dir) / "_cache"
        if cfg.llm.cache:
            llm_cache.configure(self.cache_dir / "llm")

        log(f"[init] agent: {cfg.llm.backend} ({cfg.llm.model})")
        self.agent = get_agent(cfg.llm)
        llm = getattr(self.agent, "llm", None)
        log(f"[init] perception: {cfg.perception.backend}")
        self.perceiver, self.clip = get_perceiver(cfg.perception, llm)
        log(f"[init] retrieval: {cfg.retrieval.backend} | extract: {cfg.retrieval.extract} | web_api: {cfg.retrieval.web_api}")
        self.retriever = get_retriever(cfg.retrieval, self.clip, self.cache_dir, k_images=cfg.search_viz.k_images)
        log(f"[init] t2i: {cfg.t2i.backend} ({cfg.t2i.model})" + (" | LCM vòng nhanh" if cfg.t2i.fast_iters else "")
            + ("" if cfg.review.enabled else " | review TẮT (chỉ sinh vòng 0 + CLIP)"))
        self.generator = get_generator(cfg.t2i)
        log(f"[init] judge: {cfg.judge.backend}")
        self.judge = (st_eval.AgentJudge(self.agent) if cfg.t2i.backend == "stub" or cfg.judge.backend in ("vlm", "rule")
                      else st_eval.get_judge(cfg.judge, self.agent, self.clip))
        log(f"[init] xong. run_dir = {self.run_dir}")
        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    used = torch.cuda.memory_allocated(i) / 2**30
                    total = torch.cuda.get_device_properties(i).total_memory / 2**30
                    log(f"[init] GPU {i}: {used:.1f} / {total:.1f} GB đã cấp sau khi nạp model")
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ stage 1-3
    def stages_1_to_3(self, prompt: Prompt) -> tuple[AnalysisResult, SearchResult, CulturalSpec, bool]:
        """Trả (analysis, search, spec, from_cache). Ghi file stage*.json vào run_dir/<pid>."""
        cfg, log = self.cfg, self.log
        out = self.run_dir / prompt.id
        key = stage_cache_key(cfg, prompt, self.kb.version)
        cached = None
        if cfg.cache.enabled and not cfg.cache.refresh:
            cached = load_stage_cache(self.cache_dir, key)
        if cached:
            analysis, search, spec = cached
            log(f"  [1-3] dùng cache (prompt và cấu hình không đổi): spec {', '.join(e.name_vi for e in spec.entities) or '(rỗng)'}")
            for se in spec.entities:  # thực thể ad-hoc trong cache phải được nạp lại vào KB bộ nhớ
                if self.kb.get(se.entity_id) is None:
                    ent = self.kb.add_adhoc(se.name_vi, se.name_en)
                    ent.must_have, ent.must_not, ent.confusable_with = se.required_attrs, se.forbidden_attrs, se.confusables
            st_extract.rehydrate(search, self.kb, self.cache_dir / "kb_auto")  # bản KB tự sinh (v1.8)
        else:
            analysis = st_analysis.run(self.agent, prompt, self.kb, cfg.max_candidate_entities)
            log(f"  [1] ứng viên: {analysis.candidate_entity_ids} | vùng: {analysis.region_hint or '-'}"
                + (f" | mới: {[n.name_vi for n in analysis.new_entities]}" if analysis.new_entities else ""))
            search = self.retriever.search(analysis, self.kb, raw_prompt=prompt.text_vi)
            n_img = sum(1 for i in search.items if i.kind == "image" and i.local_path)
            n_ref = sum(1 for i in search.items if i.is_reference)
            n_txt = sum(1 for i in search.items if i.kind in ("wiki_text", "web_text") and i.provenance != "kb.notes (offline)")
            log(f"  [2] {len(search.items)} bằng chứng: {n_txt} văn bản online, {n_img} ảnh tải về, {n_ref} tham chiếu đạt CLIP"
                + (f" | lỗi: {search.retrieval_errors[:2]}" if search.retrieval_errors else ""))
            search = st_extract.run(self.agent, search, self.kb, cfg.retrieval, self.cache_dir / "evidence", log=log,
                                    kb_auto_dir=self.cache_dir / "kb_auto")
            for note in search.notes[:4]:
                log(f"  [2b] {note}")
            spec = st_spec.run(self.agent, prompt, analysis, search, self.kb, cfg.max_spec_entities, cfg.min_entity_score)
            log(f"  [3] spec: {', '.join(e.name_vi for e in spec.entities) or '(rỗng)'}"
                + (f" | bỏ {len(spec.dropped)}" if spec.dropped else ""))
            if cfg.cache.enabled:
                save_stage_cache(self.cache_dir, key, prompt, analysis, search, spec)
        _write(out / "stage1_analysis.json", analysis)
        _write(out / "stage2_search.json", search)
        _write(out / "stage3_spec.json", spec)
        return analysis, search, spec, bool(cached)

    # ------------------------------------------------------------------ một prompt
    def run_one(self, prompt: Prompt) -> PipelineResult:
        out = self.run_dir / prompt.id
        cfg, log = self.cfg, self.log
        t0 = time.time()
        analysis, search, spec, _ = self.stages_1_to_3(prompt)

        if cfg.review.enabled:
            outcome = st_review.run(self.agent, self.generator, self.perceiver, self.clip, prompt, spec, self.kb,
                                    analysis.prompt_en, cfg, out, log=log)
        else:
            outcome = st_review.generate_only(self.generator, self.clip, prompt, spec, self.kb, analysis.prompt_en, cfg, out, log=log)
        _write(out / "stage45_review.json", outcome)

        record = st_eval.run(self.judge, prompt, spec, search, outcome)
        _write(out / "stage6_eval.json", record)
        log(f"  [6] {'ĐẠT' if record.passed else 'chưa đạt'} | CLIP {record.clip_fidelity:.2f} | judge {record.judge_score:.2f}"
            f" | {time.time() - t0:.0f}s | {record.final_image_path}")
        return PipelineResult(prompt, analysis, search, spec, outcome, record)

    # ------------------------------------------------------------------ batch
    def run_batch(self, prompts: list[Prompt]) -> tuple[list[PipelineResult], RunSummary]:
        t0 = time.time()
        results = []
        for i, p in enumerate(prompts, 1):
            self.log(f"\n[{i}/{len(prompts)}] {p.id} ({p.difficulty}) {p.text_vi}")
            try:
                results.append(self.run_one(p))
            except Exception as exc:  # noqa: BLE001 - một prompt lỗi không được làm hỏng cả batch
                self.log(f"  !! LỖI {type(exc).__name__}: {exc}")
                (self.run_dir / p.id).mkdir(parents=True, exist_ok=True)
                (self.run_dir / p.id / "ERROR.txt").write_text(f"{type(exc).__name__}: {exc}", encoding="utf-8")
            self._finalize(results, time.time() - t0)
        return results, self._summarize(results, time.time() - t0)

    def run_batch_multigen(self, prompts: list[Prompt], models: list[str] | None = None) -> dict[str, MultiGenResult]:
        """Nhiều model × nhiều prompt: stage 1-3 cho mọi prompt trước, rồi VÒNG NGOÀI THEO MODEL
        để mỗi model chỉ nạp một lần. Ghi runs/<run>/<pid>/multigen.json và grid.png từng prompt."""
        from .models import loader as model_loader
        from .stages import multigen as st_mg
        from .stages.generation import build_initial_spec

        models = list(models or self.cfg.models)
        prepared = []
        for p in prompts:
            self.log(f"\n[1-3] {p.id} {p.text_vi}")
            try:
                a, s, sp, _ = self.stages_1_to_3(p)
                gen = build_initial_spec(p, sp, a.prompt_en, self.cfg.t2i, self.cfg.seed, self.cfg.t2i.init_negatives)
                prepared.append((p, a, sp, gen))
            except Exception as exc:  # noqa: BLE001
                self.log(f"  !! LỖI {type(exc).__name__}: {exc}")
        # Giải phóng VLM trước khi nạp bộ sinh (1×T4).
        self.agent = None
        self.perceiver = None
        model_loader.free_vram()

        itm = None
        if self.cfg.multigen.itm and self.cfg.judge.backend == "blip2_itm" and self.cfg.t2i.backend != "stub":
            try:
                itm = st_eval.ITMJudge(self.cfg.judge, self.clip)
            except Exception as exc:  # noqa: BLE001
                self.log(f"[multigen] không nạp được BLIP-2 ITM: {type(exc).__name__}")
        results: dict[str, MultiGenResult] = {}
        lora_dir = Path(self.cfg.multigen.lora_dir) if self.cfg.multigen.lora_dir else self.cache_dir / "lora"
        for key in models:
            self.log(f"\n=== model {key} ===")
            for p, a, sp, gen in prepared:
                out = self.run_dir / p.id
                res = st_mg.run(gen, sp, self.kb, [key], self.cfg.multigen, out, clip=self.clip, itm=itm,
                                prompt_en=a.prompt_en or p.text_en, log=self.log, lora_dir=lora_dir)
                prev = results.get(p.id)
                if prev is None:
                    results[p.id] = res
                else:
                    prev.runs = [r for r in prev.runs if r.model_key != key] + res.runs
                    prev.grid_path = str(st_mg.draw_grid(prev, sp, out / "grid.png"))
                    (out / "multigen.json").write_text(json.dumps(to_dict(prev), ensure_ascii=False, indent=1), encoding="utf-8")
        return results

    def _finalize(self, results, wall):
        """Ghi tổng hợp sau MỖI prompt để phiên Kaggle bị ngắt vẫn còn kết quả."""
        if not results:
            return
        summary = self._summarize(results, wall)
        _write(self.run_dir / "summary.json", summary)
        _write(self.run_dir / "records.json", [r.record for r in results])
        st_eval.export_user_study([r.record for r in results], self.run_dir / "user_study.csv")
        st_eval.write_html_report(results, summary, self.run_dir / "report.html")

    def _summarize(self, results, wall) -> RunSummary:
        ok = [r for r in results if r.record.verifiable]
        n = max(1, len(ok))
        recs = [r.record for r in ok]
        recalls = [r.retrieval_recall for r in recs if r.retrieval_recall >= 0]
        oracles = [r.oracle_fidelity for r in recs if r.oracle_fidelity is not None]
        return RunSummary(
            run_id=self.run_id, config=self.cfg.as_dict(), n_prompts=len(results),
            n_unverifiable=len(results) - len(ok),
            pass_rate=sum(r.passed for r in recs) / n,
            pass_rate_iter0=sum(r.outcome.iterations[0].adjudication.verdict == "pass" for r in ok) / n,
            mean_iterations=sum(r.iterations for r in recs) / n,
            mean_clip_fidelity=sum(r.clip_fidelity for r in recs) / n,
            mean_clip_fidelity_iter0=sum(st_eval.clip_fidelity(r.outcome, r.spec, 0) for r in ok) / n,
            mean_retrieval_recall=(sum(recalls) / len(recalls)) if recalls else -1.0,
            mean_judge_score=sum(r.judge_score for r in recs) / n,
            mean_oracle_fidelity=(sum(oracles) / len(oracles)) if oracles else None,
            wall_seconds=wall,
        )
