"""Nối các stage. Đọc file này để thấy toàn bộ luồng trên một trang."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .config import Config
from .kb import KnowledgeBase
from .llm.base import get_agent
from .schema import (AnalysisResult, CulturalSpec, EvalRecord, Prompt, ReviewOutcome, RunSummary,
                     SearchResult, to_dict)
from .stages import analysis as st_analysis
from .stages import evaluation as st_eval
from .stages import review as st_review
from .stages import spec as st_spec
from .stages.generation import get_generator
from .stages.perception import get_perceiver
from .stages.retrieval import get_retriever


def load_prompts(path: str | Path) -> list[Prompt]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("//"):
            out.append(Prompt(**json.loads(line)))
    return out


@dataclass
class PipelineResult:
    prompt: Prompt
    analysis: AnalysisResult
    search: SearchResult
    spec: CulturalSpec
    outcome: ReviewOutcome
    record: EvalRecord


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_dict(obj), ensure_ascii=False, indent=2), encoding="utf-8")


class Pipeline:
    def __init__(self, cfg: Config, log=print):
        self.cfg, self.log = cfg, log
        self.kb = KnowledgeBase.load(cfg.kb_path)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = cfg.run_name or f"{stamp}-{cfg.llm.backend}-{cfg.t2i.backend}"
        self.run_dir = Path(cfg.runs_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        _write(self.run_dir / "config.json", cfg.as_dict())

        log(f"[init] agent: {cfg.llm.backend} ({cfg.llm.model})")
        self.agent = get_agent(cfg.llm)
        llm = getattr(self.agent, "llm", None)
        log(f"[init] perception: {cfg.perception.backend}")
        self.perceiver, self.clip = get_perceiver(cfg.perception, llm)
        log(f"[init] retrieval: {cfg.retrieval.backend}")
        self.retriever = get_retriever(cfg.retrieval, self.clip, Path(cfg.runs_dir) / "_cache" / "ref_images")
        log(f"[init] t2i: {cfg.t2i.backend} ({cfg.t2i.model})")
        self.generator = get_generator(cfg.t2i)
        log(f"[init] xong. run_dir = {self.run_dir}")

    def run_one(self, prompt: Prompt) -> PipelineResult:
        out = self.run_dir / prompt.id
        cfg, log = self.cfg, self.log
        t0 = time.time()

        analysis = st_analysis.run(self.agent, prompt, self.kb)
        _write(out / "stage1_analysis.json", analysis)
        log(f"  [1] ứng viên: {analysis.candidate_entity_ids} | vùng: {analysis.region_hint or '-'}")

        search = self.retriever.search(analysis, self.kb)
        _write(out / "stage2_search.json", search)
        n_img = sum(1 for i in search.items if i.kind == "image" and i.local_path)
        log(f"  [2] {len(search.items)} bằng chứng, {n_img} ảnh tham chiếu đạt CLIP"
            + (f" | lỗi mạng: {len(search.retrieval_errors)}" if search.retrieval_errors else ""))

        spec = st_spec.run(self.agent, prompt, analysis, search, self.kb, cfg.max_spec_entities, cfg.min_entity_score)
        _write(out / "stage3_spec.json", spec)
        log(f"  [3] spec: {', '.join(e.name_vi for e in spec.entities) or '(rỗng)'}"
            + (f" | bỏ {len(spec.dropped)}" if spec.dropped else ""))

        outcome = st_review.run(self.agent, self.generator, self.perceiver, self.clip, prompt, spec, self.kb,
                                analysis.prompt_en, cfg, out, log=log)
        _write(out / "stage45_review.json", outcome)

        record = st_eval.run(self.agent, prompt, spec, search, outcome)
        _write(out / "stage6_eval.json", record)
        log(f"  [6] {'ĐẠT' if record.passed else 'chưa đạt'} | CLIP {record.clip_fidelity:.2f} | judge {record.judge_score:.2f}"
            f" | {time.time() - t0:.0f}s | {record.final_image_path}")
        return PipelineResult(prompt, analysis, search, spec, outcome, record)

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
