"""
Session: các bước pipeline dưới dạng method có memo, dùng trong notebook.

Chạy lại một cell khi đầu vào không đổi -> trả kết quả cũ (bộ nhớ, rồi đĩa), không gọi API,
VLM hay bộ sinh. Đổi prompt/config -> chỉ bước đó và các bước sau chạy lại. Mỗi kết quả kèm
`source` ("memory" | "disk" | "computed") để viz hiện badge.

Model nặng (agent, CLIP, BLIP-2) giữ trong Session; `free_vlm()` giải phóng trước bước sinh ảnh
trên 1×T4.

    s = Session(cfg, prompt)
    a, src = s.analysis();      viz.show(viz.keywords_table(a, s.kb, source=src))
    cmp, src = s.compare();     viz.show(viz.query_comparison(cmp, source=src))
    search, src = s.retrieve()
    spec, src = s.spec();       gen, _ = s.genspec()
    s.free_vlm()
    res, src = s.multigen(cfg.models, on_model_done=...)
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import Config
from .kb import KnowledgeBase
from .llm import cache as llm_cache
from .schema import (
    CandidateReview, CulturalBrief, FilterResult,
    AnalysisResult, CulturalSpec, GenSpec, MultiGenResult, Prompt, QueryComparison, ReviewOutcome, SearchResult,
    from_dict, to_dict,
)


#: Phiên bản LOGIC của từng bước. Tăng số khi đổi code làm đầu ra bước khác đi dù đầu vào không đổi,
#: để cache bước cũ trên đĩa (step_*.json) không che mất thay đổi. Các bước sau tự đổi khoá vì khoá
#: của chúng chứa hash đầu ra bước trước.
STEP_LOGIC = {"analysis": 2, "compare": 1, "retrieve": 2, "spec": 1, "genspec": 4, "multigen": 3, "review": 1,
              "brief": 2, "ref_filter": 2, "candidate_review": 2}


def _h(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()[:12]


@dataclass
class _Step:
    value: Any = None
    key: str = ""
    source: str = "computed"


@dataclass
class Session:
    cfg: Config
    prompt: Prompt
    run_dir: Path | None = None
    log: Callable[[str], None] = print
    kb: KnowledgeBase = field(init=False)
    steps: dict[str, _Step] = field(default_factory=dict)

    # model nặng, nạp lười
    _agent: Any = None
    _clip: Any = None
    _itm: Any = None
    _aesthetic: Any = None
    _web: Any = None

    def __post_init__(self):
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        self.kb = KnowledgeBase.load(self.cfg.kb_path)
        self.cache_dir = Path(self.cfg.cache.dir) if self.cfg.cache.dir else Path(self.cfg.runs_dir) / "_cache"
        if self.run_dir is None:
            self.run_dir = Path(self.cfg.runs_dir) / (self.cfg.run_name or "walkthrough")
        self.out_dir = Path(self.run_dir) / self.prompt.id
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg.llm.cache:
            llm_cache.configure(self.cache_dir / "llm")

    # ------------------------------------------------------------------ hạ tầng memo
    def _memo(self, name: str, key: str, cls, compute: Callable[[], Any], force: bool = False, reusable=None):
        """`reusable(val) -> bool`: kết quả cũ có đáng dùng lại không. Mặc định có; multigen từ chối kết quả
        có hàng lỗi (v1.2 p001: LoraError bị đóng băng trong step_multigen.json nên sửa môi trường xong vẫn thấy lỗi)."""
        ok = reusable or (lambda v: True)
        key = f"{key}-L{STEP_LOGIC.get(name, 1)}"
        st = self.steps.get(name)
        if st and st.key == key and not force and ok(st.value):
            st.source = "memory"
            return st.value, "memory"
        path = self.out_dir / f"step_{name}.json"
        if not force and self.cfg.cache.enabled and not self.cfg.cache.refresh and path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if raw.get("_key") == key:
                    val = from_dict(cls, raw["value"]) if cls else raw["value"]
                    if ok(val):
                        self.steps[name] = _Step(val, key, "disk")
                        return val, "disk"
                    self.log(f"  [{name}] cache đĩa có hàng lỗi/thiếu -> chạy lại bước (ảnh của hàng tốt vẫn tái dùng)")
            except Exception:  # noqa: BLE001
                pass
        val = compute()
        self.steps[name] = _Step(val, key, "computed")
        try:
            path.write_text(json.dumps({"_key": key, "value": to_dict(val)}, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
        return val, "computed"

    def invalidate(self, name: str) -> None:
        """Xoá bước `name` và mọi bước sau nó (trong bộ nhớ và trên đĩa)."""
        order = ["analysis", "compare", "retrieve", "brief", "spec", "genspec", "ref_filter", "multigen", "candidate_review", "review"]
        if name not in order:
            return
        for n in order[order.index(name):]:
            self.steps.pop(n, None)
            p = self.out_dir / f"step_{n}.json"
            if p.exists():
                p.unlink()

    # ------------------------------------------------------------------ model nặng
    @property
    def agent(self):
        if self._agent is None:
            from dataclasses import replace as _replace

            from .llm.base import get_agent

            llm_cfg = self.cfg.llm
            if llm_cfg.backend == "qwen_vl":
                dev = self._pick_vlm_device(llm_cfg.device, need_gb=8.0)
                if dev != llm_cfg.device:
                    llm_cfg = _replace(llm_cfg, device=dev)
            self.log(f"[session] nạp agent {llm_cfg.backend}" + (f" ({llm_cfg.model}, {llm_cfg.device})" if llm_cfg.backend != "rule" else ""))
            self._agent = get_agent(llm_cfg)
        return self._agent

    def _pick_vlm_device(self, preferred: str, need_gb: float) -> str:
        """v1.5.2: nạp lại VLM ở bước 4c bị OOM khi GPU 0 còn giữ Qwen của Session cũ + CLIP + OWL-ViT. Dọn cache rồi
        chọn GPU còn >= need_gb trống (2xT4: GPU 1 rảnh sau bước 4); không GPU nào đủ thì vẫn trả preferred và ghi cảnh báo."""
        from .models.loader import free_gb, free_vram

        if not str(preferred).startswith("cuda"):
            return preferred
        free_vram()
        try:
            import torch

            n = torch.cuda.device_count()
        except Exception:  # noqa: BLE001
            return preferred
        cands = [preferred] + [f"cuda:{i}" for i in range(n) if f"cuda:{i}" != preferred]
        frees = {d: free_gb(d) for d in cands}
        for d in cands:
            if frees.get(d) is not None and frees[d] >= need_gb:
                if d != preferred:
                    self.log(f"[session] {preferred} còn {frees[preferred]} GB < {need_gb} GB -> nạp VLM lên {d} ({frees[d]} GB trống)")
                return d
        self.log(f"[session] CẢNH BÁO: không GPU nào còn >= {need_gb} GB trống ({frees}); nạp VLM lên {preferred}, có thể OOM. "
                 "Restart kernel (cache đĩa giữ mọi bước) là cách chắc nhất.")
        return preferred

    @property
    def clip(self):
        if self._clip is None and self.cfg.perception.backend != "stub":
            from .stages.perception import CLIPProbe

            self._clip = CLIPProbe(self.cfg.perception.clip_model, self.cfg.perception.device)
        return self._clip

    @property
    def itm(self):
        if self._itm is None and self.cfg.multigen.itm and self.cfg.judge.backend == "blip2_itm" and self.cfg.t2i.backend != "stub":
            from .stages.evaluation import ITMJudge

            try:
                self._itm = ITMJudge(self.cfg.judge, self.clip)
            except Exception as exc:  # noqa: BLE001
                self.log(f"[session] không nạp được BLIP-2 ITM ({type(exc).__name__}); bỏ điểm ITM")
                self._itm = False
        return self._itm or None

    _ref_index: Any = None

    @property
    def ref_index(self):
        """Kho ảnh tham chiếu CLIP (v1.6). None nếu không cấu hình hoặc không nạp được."""
        if self._ref_index is None:
            self._ref_index = False
            path = self.cfg.retrieval.ref_index
            if path and Path(path).exists():
                try:
                    from .stages.refindex import RefIndex

                    idx = RefIndex.load(path)
                    if idx.clip_model and idx.clip_model != self.cfg.perception.clip_model:
                        self.log(f"[session] kho ảnh đánh chỉ mục bằng {idx.clip_model}, đang dùng {self.cfg.perception.clip_model} -> bỏ kho")
                    else:
                        self._ref_index = idx
                        self.log(f"[session] kho ảnh tham chiếu: {len(idx)} ảnh ({path})")
                except Exception as exc:  # noqa: BLE001
                    self.log(f"[session] không nạp được kho ảnh {path}: {type(exc).__name__}: {exc}")
            elif path:
                self.log(f"[session] kho ảnh {path} không tồn tại -> dùng web search")
        return self._ref_index or None

    def index_refs(self, captions: list[str], k: int | None = None) -> list[tuple[str, float]]:
        """Truy hồi từ kho theo các caption (hợp, giữ điểm cao nhất mỗi ảnh), cosine >= ref_index_min_sim."""
        idx = self.ref_index
        if idx is None or not captions:
            return []
        k = k or self.cfg.retrieval.ref_index_k
        best: dict[str, float] = {}
        try:
            vecs = self.clip.text_embed(captions)
            for v in vecs:
                for pth, sim in idx.search(v.cpu().numpy() if hasattr(v, "cpu") else v, k=k, min_sim=self.cfg.retrieval.ref_index_min_sim):
                    best[pth] = max(best.get(pth, 0.0), sim)
        except Exception as exc:  # noqa: BLE001
            self.log(f"  [3b] kho ảnh lỗi: {type(exc).__name__}: {str(exc)[:80]}")
            return []
        return sorted(best.items(), key=lambda kv: -kv[1])[:k]

    def attribute_refs(self, missing_attrs: list[str], k: int = 3) -> list[str]:
        """v1.6 (ImageRAG): vòng sửa truy hồi ảnh theo CAPTION của THUỘC TÍNH thiếu ("close-up of a Vietnamese ao dai showing
        a high stand-up collar"), không theo tên thực thể. Kho trước, web (DDG ảnh) sau; lọc CLIP theo caption; cắt theo thực thể."""
        sp, _ = self.spec()
        main = next((se for se in sp.entities if se.kind == "object"), None)
        if main is None or not missing_attrs:
            return []
        name = main.name_en.split("(")[0].strip()
        caps = [f"close-up photo of a Vietnamese {name} showing {a}" for a in missing_attrs[:3]]
        hits = self.index_refs(caps, k=k * 2)
        src = "kho"
        if not hits:
            src = "web"
            thr = self.cfg.retrieval.ref_index_min_sim
            for cap in caps[:2]:
                try:
                    for r in self.web.images(cap, n=4):
                        local = self.web.download(r.get("image"))
                        if not local:
                            continue
                        sim = self.clip.similarity(local, [cap])[0]
                        if sim >= thr:
                            hits.append((local, sim))
                except Exception as exc:  # noqa: BLE001
                    self.log(f"  [4d] web ảnh cho '{cap[:40]}': {type(exc).__name__}")
            hits = sorted(dict(hits).items(), key=lambda kv: -kv[1])
        paths = [p for p, _ in hits][:k]
        self.log(f"  [4d] ảnh theo caption thuộc tính ({src}): {len(paths)} ảnh cho {[a[:30] for a in missing_attrs[:3]]}")
        if paths and self.cfg.multigen.ref_crop:
            paths = self.crop_refs(paths, sp)
        return paths

    @property
    def aesthetic(self):
        """PickScore (v1.3), nạp lười, offload CPU. None nếu tắt hoặc không nạp được."""
        if self._aesthetic is None:
            if self.cfg.t2i.backend == "stub" or not self.cfg.multigen.aesthetic.enabled:
                self._aesthetic = False
            else:
                from .stages.aesthetic import get_scorer

                self._aesthetic = get_scorer(self.cfg.multigen.aesthetic, log=self.log) or False
        return self._aesthetic or None

    def reference_images(self, k: int | None = None) -> list[str]:
        """Ảnh tham chiếu cho IP-Adapter, chọn theo TẦNG (v1.5.3):
          1. ảnh của thực thể vật thể, CLIP >= ref_image_min_clip (0,75; áo dài đạt dễ);
          2. như trên nhưng ngưỡng nới (>= 0,5): thực thể hiếm (thuyền thúng) CLIP kém tự tin dù ảnh đúng;
          3. ảnh từ nhóm truy vấn PROMPT GỐC (entity "-"), chấm lại bằng CLIP theo nhãn thực thể so với confusable, >= 0,5.
        Xếp theo P(thực thể) + độ khớp prompt (số người, bố cục). Rỗng thì log rõ số ảnh đã xét và điểm cao nhất.
        Sau đó Filter agent lọc và cắt theo thực thể."""
        s, _ = self.retrieve()
        sp, _ = self.spec()
        objs = [se for se in sp.entities if se.kind == "object"]
        obj_ids = {se.entity_id for se in objs}
        thr = self.cfg.retrieval.ref_image_min_clip
        imgs = [it for it in s.items if it.kind == "image" and it.local_path and Path(it.local_path).exists()]
        # Tầng 0 (v1.6): kho ảnh của nhóm đánh chỉ mục CLIP, truy hồi bằng nhãn thực thể + prompt EN.
        if self.ref_index is not None and objs:
            main = max(objs, key=lambda se: se.weight)
            a0, _ = self.analysis()
            caps = [main.clip_label or f"a photo of Vietnamese {main.name_en.split('(')[0].strip()}", a0.prompt_en or self.prompt.text_en]
            hits = self.index_refs([c for c in caps if c])
            if hits:
                k0 = k or self.cfg.multigen.ref_images
                out0 = [p for p, _ in hits]
                self.log(f"  [3b] ảnh tham chiếu tầng 0 (kho, cosine cao nhất {hits[0][1]:.2f}): {len(out0)} ảnh")
                ag0 = self.cfg.agents
                if ag0.enabled and ag0.ref_filter:
                    flt, _ = self.ref_filter(out0[: max(k0 * 2, 4)])
                    kept = [p for p in out0 if p in set(flt.kept)]
                    if kept:
                        out0 = kept
                out0 = out0[:max(1, k0)]
                return self.crop_refs(out0, sp) if self.cfg.multigen.ref_crop else out0
        ent_imgs = [it for it in imgs if it.entity_id in obj_ids]
        tier1 = [it for it in ent_imgs if it.clip_match is not None and it.clip_match >= thr]
        tier2 = [it for it in ent_imgs if it.clip_match is not None and 0.5 <= it.clip_match < thr]
        chosen, tier = tier1, "1 (CLIP >= %.2f)" % thr
        if not chosen and tier2:
            chosen, tier = tier2, "2 (CLIP >= 0,50, ngưỡng nới cho thực thể hiếm)"
        if not chosen and objs:
            main = max(objs, key=lambda se: se.weight)
            label = main.clip_label or f"a photo of Vietnamese {main.name_en.split('(')[0].strip()}"
            from .llm.shared import confusable_clip_label

            cfs = [confusable_clip_label(c) for c in main.confusables[:4]]
            prompt_imgs = [it for it in imgs if it.entity_id == "-"]
            scored = []
            for it in prompt_imgs:
                try:
                    m = self.clip.image_matches(it.local_path, label, cfs)
                except Exception:  # noqa: BLE001
                    continue
                it.clip_match = round(m, 4)
                if m >= 0.5:
                    scored.append(it)
            if scored:
                chosen, tier = scored, f"3 (ảnh từ prompt gốc, chấm lại theo nhãn '{label[:40]}')"
        if not chosen:
            best = max([it.clip_match for it in ent_imgs if it.clip_match is not None] + [0.0])
            self.log(f"  [3b] không có ảnh tham chiếu: {len(ent_imgs)} ảnh thực thể (CLIP cao nhất {best:.2f}), "
                     f"{len([it for it in imgs if it.entity_id == '-'])} ảnh prompt gốc đều dưới 0,50")
            return []
        a, _ = self.analysis()
        pe = a.prompt_en or self.prompt.text_en

        def rank(it):
            sim = 0.0
            if pe:
                try:
                    sim = self.clip.similarity(it.local_path, [pe])[0]
                except Exception:  # noqa: BLE001
                    sim = 0.0
            return -(0.5 * (it.clip_match or 0) + sim)

        chosen = sorted(chosen, key=rank)
        out: list[str] = []
        for it in chosen:
            if it.local_path not in out:
                out.append(it.local_path)
        self.log(f"  [3b] ảnh tham chiếu tầng {tier}: {len(out)} ảnh")
        k = k or self.cfg.multigen.ref_images
        ag = self.cfg.agents
        if ag.enabled and ag.ref_filter and out:
            flt, _ = self.ref_filter(out[: max(k * 2, 4)])
            kept = [p for p in out if p in set(flt.kept)]
            if kept:
                out = kept
        out = out[:max(1, k)]
        if self.cfg.multigen.ref_crop and out:
            out = self.crop_refs(out, sp)
        return out

    def crop_refs(self, paths: list[str], sp) -> list[str]:
        """v1.4.2: cắt từng ảnh tham chiếu về vùng thực thể vật thể chính bằng CLIP quét lưới (cache theo hash)."""
        from .stages.refcrop import crop_to_entity

        main = next((se for se in sp.entities if se.kind == "object"), None)
        if main is None or not hasattr(self.clip, "similarity_image"):
            return paths
        label = main.clip_label or f"a photo of Vietnamese {main.name_en.split('(')[0].strip()}"
        out = []
        n_crop = 0
        for p in paths:
            try:
                q, info = crop_to_entity(self.clip, p, label, self.cache_dir / "ref_crops",
                                         detector=self.cfg.multigen.ref_detector, device=self.cfg.perception.device)
                n_crop += int(bool(info.get("cropped")) or (info.get("cached") and q != p))
                out.append(q)
            except Exception as exc:  # noqa: BLE001
                self.log(f"  [3c] không cắt được {Path(p).name}: {type(exc).__name__}: {str(exc)[:60]}")
                out.append(p)
        self.log(f"  [3c] ảnh tham chiếu: cắt {n_crop}/{len(paths)} về vùng '{label[:50]}' ({self.cfg.multigen.ref_detector})")
        return out

    # ------------------------------------------------------------------ v1.4 agents
    def brief(self, force: bool = False) -> tuple[dict, str]:
        """Bước 2c - Summary agent: mỗi thực thể trong spec một CulturalBrief (facts thị giác, khác gì confusable)."""
        from .agents import summary as ag_sum

        s, _ = self.retrieve()
        sp, _ = self.spec()
        key = _h({**self._base_key(), "spec": _h(to_dict(sp)), "search": _h([it.url for it in s.items])})

        def compute():
            if not (self.cfg.agents.enabled and self.cfg.agents.summary):
                return {}
            return {k: to_dict(v) for k, v in ag_sum.run(self.agent, s, sp, self.kb, log=self.log).items()}

        val, src = self._memo("brief", key, None, compute, force)
        return {k: from_dict(CulturalBrief, v) for k, v in (val or {}).items()}, src

    def ref_filter(self, paths: list[str], force: bool = False) -> tuple[FilterResult, str]:
        """Bước 3b - Filter agent trên ảnh tham chiếu (trước IP-Adapter)."""
        from .agents import describe as ag_desc

        sp, _ = self.spec()
        a, _ = self.analysis()
        pe = a.prompt_en or self.prompt.text_en
        key = _h({"paths": [Path(p).name for p in paths], "spec": _h(to_dict(sp)), "pe": pe})
        return self._memo("ref_filter", key, FilterResult,
                          lambda: ag_desc.run(self.agent, paths, sp, pe, kind="reference", log=self.log, clip=self.clip), force)

    def candidate_review(self, force: bool = False) -> tuple[CandidateReview, str]:
        """Bước 4c - Filter + Rank trên top-k ứng viên multigen, rồi (tuỳ chọn) một vòng sửa + sinh lại."""
        from .agents import describe as ag_desc, loop as ag_loop, rank as ag_rank
        from .stages.multigen import combined_score

        res, _ = self.multigen()
        sp, _ = self.spec()
        gen, _ = self.genspec()
        a, _ = self.analysis()
        briefs, _ = self.brief()
        c = self.cfg.agents
        pe = a.prompt_en or self.prompt.text_en
        seen: set[str] = set()
        cands = []
        for cand, m in sorted([(cand, r.model_key) for r in res.runs if r.output for cand in r.output.candidates],
                              key=lambda cm: -combined_score(cm[0])):
            if cand.path in seen:
                continue  # hàng alias dùng lại ảnh của hàng gốc -> không chấm hai lần
            seen.add(cand.path)
            cands.append((cand, m))
        cands = cands[: c.k_candidates]
        key = _h({"paths": [Path(cand.path).name for cand, _ in cands], "spec": _h(to_dict(sp)), "pe": pe,
                  "k": c.k_candidates, "rev": c.max_revisions, "gen": _h(to_dict(gen))})

        def compute():
            flt = ag_desc.run(self.agent, [cand.path for cand, _ in cands], sp, pe, kind="candidate", log=self.log, clip=self.clip)
            rk = ag_rank.run(self.agent, cands, flt, briefs, sp, pe, log=self.log)
            best = rk.final_order[0] if rk.final_order else None
            model_of = {cand.path: m for cand, m in cands}
            cr = CandidateReview(prompt_id=self.prompt.id, k=len(cands), filter=flt, rank=rk, best_path=best,
                                 best_model=model_of.get(best) if best else None, final_path=best)
            v0 = next((v for v in flt.verdicts if v.path == best), None)
            if c.max_revisions > 0 and best and ag_loop.needs_revision(v0):
                plan = ag_loop.plan_from_verdict(v0, sp, gen)
                cr.revision = plan
                self.log(f"  [4d] sửa: {plan.rationale} -> +{plan.add_positive} -{plan.add_negative} boost={plan.boost} g+{plan.guidance_delta}")
                if plan.is_empty():
                    cr.notes.append("kế hoạch sửa rỗng (không còn gì để thêm) -> giữ ảnh multigen")
                    return cr
                refs_for_regen = self.attribute_refs(v0.missing_must_have) if plan.use_reference_image else []
                if not refs_for_regen:
                    refs_for_regen = self.reference_images()
                else:
                    cr.notes.append(f"vòng sửa dùng {len(refs_for_regen)} ảnh truy hồi theo caption thuộc tính thiếu")
                run_rec, _ = ag_loop.regenerate(gen, plan, sp, self.kb, cr.best_model, self.cfg, self.out_dir,
                                                clip=self.clip, itm=self.itm, prompt_en=pe, log=self.log,
                                                aesthetic=self.aesthetic, ref_images=refs_for_regen)
                cr.regen = run_rec
                if run_rec is not None and run_rec.output:
                    flt2 = ag_desc.run(self.agent, [x.path for x in run_rec.output.candidates], sp, pe, kind="candidate", log=self.log, clip=self.clip)
                    cr.regen_filter = flt2
                    good = [v for v in flt2.verdicts if v.keep and not v.matched_must_not]
                    pick = max(good, key=lambda v: v.score) if good else None
                    if pick is not None and pick.score > v0.score:  # chỉ đổi khi tốt hơn THẬT, hoà thì giữ ảnh gốc
                        cr.final_path, cr.final_source = pick.path, "regen"
                        cr.notes.append(f"vòng sửa cho ảnh sạch hơn ({pick.score:+.2f} so với {v0.score:+.2f})")
                    else:
                        cr.notes.append("vòng sửa không tốt hơn ảnh multigen" + (f" ({pick.score:+.2f} so với {v0.score:+.2f})" if pick else "") + " -> giữ ảnh multigen")
                elif run_rec is not None:
                    cr.notes.append(f"sinh lại lỗi: {run_rec.error}")
            elif v0 is not None:
                cr.notes.append("ứng viên đầu đạt: không cần vòng sửa")
            return cr

        return self._memo("candidate_review", key, CandidateReview, compute, force)

    @property
    def web(self):
        if self._web is None:
            from .stages.websearch import WebClient

            self._web = WebClient(self.cfg.retrieval, self.cache_dir)
        return self._web

    def free_vlm(self) -> None:
        """Giải phóng agent VLM trước bước sinh ảnh (1×T4). Kết quả các bước đã memo vẫn còn."""
        from .models.loader import free_vram

        if self._agent is not None:
            self._agent = None
        free_vram()
        self.log("[session] đã giải phóng VLM")

    # ------------------------------------------------------------------ các bước
    def _base_key(self) -> dict:
        c = self.cfg
        return {"text": self.prompt.text_vi, "en": self.prompt.text_en, "llm": [c.llm.backend, c.llm.model],
                "kb": self.kb.version, "maxc": c.max_candidate_entities}

    def analysis(self, force: bool = False) -> tuple[AnalysisResult, str]:
        from .stages import analysis as st

        key = _h(self._base_key())
        return self._memo("analysis", key, AnalysisResult,
                          lambda: st.run(self.agent, self.prompt, self.kb, self.cfg.max_candidate_entities), force)

    def compare(self, force: bool = False) -> tuple[QueryComparison, str]:
        from .stages.websearch import compare_queries

        a, _ = self.analysis()
        c = self.cfg
        key = _h({**self._base_key(), "ids": a.candidate_entity_ids, "web": [c.retrieval.web_api, c.retrieval.web_langs],
                  "k": [c.search_viz.k_text, c.search_viz.k_images, c.search_viz.max_entities]})
        return self._memo("compare", key, QueryComparison,
                          lambda: compare_queries(self.web, a, self.prompt, self.kb, self.clip,
                                                  c.search_viz.k_text, c.search_viz.k_images, c.search_viz.max_entities,
                                                  c.retrieval.web_langs), force)

    def retrieve(self, force: bool = False) -> tuple[SearchResult, str]:
        from .stages import extraction as st_extract
        from .stages.retrieval import get_retriever

        a, _ = self.analysis()
        c = self.cfg
        key = _h({**self._base_key(), "ids": a.candidate_entity_ids,
                  "ret": [c.retrieval.backend, c.retrieval.web_api, c.retrieval.extract, c.retrieval.wiki_chars,
                          c.retrieval.ref_image_min_clip, c.search_viz.k_images]})

        def compute():
            retriever = get_retriever(c.retrieval, self.clip, self.cache_dir, web=self.web, k_images=c.search_viz.k_images)
            s = retriever.search(a, self.kb, raw_prompt=self.prompt.text_vi)
            return st_extract.run(self.agent, s, self.kb, c.retrieval, self.cache_dir / "evidence", log=self.log)

        val, src = self._memo("retrieve", key, SearchResult, compute, force)
        if src == "disk":
            # thực thể ad-hoc và thuộc tính rút thêm phải nạp lại vào KB bộ nhớ
            for it in val.items:
                if it.provenance == "extracted":
                    ent = self.kb.get(it.entity_id)
                    if ent is not None and not ent.must_have:
                        ent.must_have, ent.must_not, ent.confusable_with = list(it.must_have), list(it.must_not), list(it.confusable_with)
        return val, src

    def spec(self, force: bool = False) -> tuple[CulturalSpec, str]:
        from .stages import spec as st_spec

        a, _ = self.analysis()
        s, _ = self.retrieve()
        c = self.cfg
        key = _h({**self._base_key(), "search": _h(to_dict(s)), "spec": [c.max_spec_entities, c.min_entity_score]})
        val, src = self._memo("spec", key, CulturalSpec,
                              lambda: st_spec.run(self.agent, self.prompt, a, s, self.kb, c.max_spec_entities, c.min_entity_score), force)
        for se in val.entities:
            if self.kb.get(se.entity_id) is None:
                ent = self.kb.add_adhoc(se.name_vi, se.name_en)
                ent.must_have, ent.must_not, ent.confusable_with = se.required_attrs, se.forbidden_attrs, se.confusables
        return val, src

    def genspec(self, force: bool = False) -> tuple[GenSpec, str]:
        from .stages.generation import build_initial_spec

        a, _ = self.analysis()
        sp, _ = self.spec()
        c = self.cfg
        n = c.multigen.n_candidates if c.multigen.enabled else c.t2i.n_candidates
        key = _h({"spec": _h(to_dict(sp)), "pe": a.prompt_en, "t2i": [c.t2i.steps, c.t2i.guidance, c.t2i.width, c.t2i.height,
                                                                          n, c.t2i.init_negatives, c.t2i.attrs_in_prompt, c.t2i.render,
                                                                          c.t2i.emphasis_weight], "seed": c.seed,
                  "enrich": bool(c.agents.enabled and c.agents.summary and c.agents.enrich_prompt)})

        def compute():
            g = build_initial_spec(self.prompt, sp, a.prompt_en, c.t2i, c.seed, c.t2i.init_negatives)
            g.n_candidates = n  # thẻ GenSpec hiện đúng số ứng viên multigen sẽ sinh (v1.3: thẻ ghi 2, grid ra 4)
            if c.agents.enabled and c.agents.summary and c.agents.enrich_prompt:
                from .agents.summary import enrich_terms

                briefs, _ = self.brief()
                extra = [x for x in enrich_terms(briefs, sp) if x not in g.prompt_terms]
                if extra:
                    g.prompt_terms = g.prompt_terms[:-1] + extra + g.prompt_terms[-1:]  # trước STYLE_SUFFIX
            return g

        return self._memo("genspec", key, GenSpec, compute, force)

    def multigen(self, models: list[str] | None = None, on_model_done=None, force: bool = False) -> tuple[MultiGenResult, str]:
        from .stages import multigen as st_mg

        models = list(models or self.cfg.models)
        gen, _ = self.genspec()
        sp, _ = self.spec()
        a, _ = self.analysis()
        c = self.cfg
        key = _h({"gen": st_mg.genspec_hash(gen, st_mg.render_settings(c.multigen)), "models": models,
                  "n": c.multigen.n_candidates, "side": c.multigen.max_side, "aes": c.multigen.aesthetic.enabled,
                  "reffilter": bool(c.agents.enabled and c.agents.ref_filter), "refcrop": c.multigen.ref_crop,
                  "refscale": c.multigen.ref_scale, "detector": c.multigen.ref_detector,
                  "adaptive": to_dict(c.multigen.adaptive), "autoref": to_dict(c.multigen.auto_ref), "ens": c.multigen.ensemble,
                  "sat": c.multigen.saturated_metrics,
                  "ov": c.multigen.overrides})
        lora_dir = Path(c.multigen.lora_dir) if c.multigen.lora_dir else self.cache_dir / "lora"

        def compute():
            refs = self.reference_images() if any(get_model(m).ip_adapter for m in models if _known(m)) else []
            return st_mg.run(gen, sp, self.kb, models, c.multigen, self.out_dir, clip=self.clip, itm=self.itm,
                             ref_images=refs, aesthetic=self.aesthetic, t2i_cfg=c.t2i,
                             prompt_en=a.prompt_en or self.prompt.text_en, log=self.log, on_model_done=on_model_done,
                             lora_dir=lora_dir)

        def reusable(v: MultiGenResult) -> bool:
            # hàng "bỏ qua: ..." (only_if_entity) là chủ ý, rẻ, không cần chạy lại; hàng lỗi thật thì phải thử lại
            return bool(v.runs) and all(r.output is not None or (r.error or "").startswith("bỏ qua") for r in v.runs)

        from .models.registry import get as get_model

        def _known(m: str) -> bool:
            try:
                get_model(m); return True
            except KeyError:
                return False

        val, src = self._memo("multigen", key, MultiGenResult, compute, force, reusable=reusable)
        if src != "computed" and on_model_done:
            for r in val.runs:
                on_model_done(r)
        return val, src

    def review(self, force: bool = False) -> tuple[ReviewOutcome, str]:
        """Vòng review agent cũ trên bộ sinh t2i (tuỳ chọn, tốn VLM + SDXL)."""
        from .stages import review as st_review
        from .stages.generation import get_generator
        from .stages.perception import get_perceiver

        a, _ = self.analysis()
        sp, _ = self.spec()
        c = self.cfg
        key = _h({"spec": _h(to_dict(sp)), "t2i": to_dict(c.t2i), "review": to_dict(c.review), "seed": c.seed})

        def compute():
            perceiver, _ = get_perceiver(c.perception, getattr(self.agent, "llm", None))
            generator = get_generator(c.t2i)
            try:
                return st_review.run(self.agent, generator, perceiver, self.clip, self.prompt, sp, self.kb,
                                     a.prompt_en, c, self.out_dir / "review", log=self.log)
            finally:
                from .models.loader import free_vram

                del generator
                free_vram()

        return self._memo("review", key, ReviewOutcome, compute, force)
