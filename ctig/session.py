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
STEP_LOGIC = {"analysis": 1, "compare": 1, "retrieve": 2, "spec": 1, "genspec": 4, "multigen": 3, "review": 1,
              "brief": 1, "ref_filter": 1, "candidate_review": 1}


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
            from .llm.base import get_agent

            self.log(f"[session] nạp agent {self.cfg.llm.backend}" + (f" ({self.cfg.llm.model})" if self.cfg.llm.backend != "rule" else ""))
            self._agent = get_agent(self.cfg.llm)
        return self._agent

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
        """Ảnh tham chiếu cho IP-Adapter: ảnh vật thể đã tải, CLIP >= ngưỡng, tốt nhất trước (v1.3: nhiều ảnh)."""
        s, _ = self.retrieve()
        sp, _ = self.spec()
        obj_ids = {se.entity_id for se in sp.entities if se.kind == "object"}
        thr = self.cfg.retrieval.ref_image_min_clip
        items = [it for it in s.items if it.kind == "image" and it.local_path and it.entity_id in obj_ids
                 and it.clip_match is not None and it.clip_match >= thr]
        # v1.3 p001: ảnh tham chiếu là ảnh NHÓM nữ sinh -> IP-Adapter Plus kéo ra 3-4 người dù prompt "một cô gái".
        # Xếp theo P(thực thể) + độ khớp prompt (bố cục, số người) thay vì chỉ P(thực thể).
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
        items = [it for it in items if Path(it.local_path).exists()]
        items.sort(key=rank)
        out: list[str] = []
        for it in items:
            if it.local_path not in out:
                out.append(it.local_path)
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
                          lambda: ag_desc.run(self.agent, paths, sp, pe, kind="reference", log=self.log), force)

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
        cands = sorted([(cand, r.model_key) for r in res.runs if r.output for cand in r.output.candidates],
                       key=lambda cm: -combined_score(cm[0]))[: c.k_candidates]
        key = _h({"paths": [Path(cand.path).name for cand, _ in cands], "spec": _h(to_dict(sp)), "pe": pe,
                  "k": c.k_candidates, "rev": c.max_revisions, "gen": _h(to_dict(gen))})

        def compute():
            flt = ag_desc.run(self.agent, [cand.path for cand, _ in cands], sp, pe, kind="candidate", log=self.log)
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
                run_rec, _ = ag_loop.regenerate(gen, plan, sp, self.kb, cr.best_model, self.cfg, self.out_dir,
                                                clip=self.clip, itm=self.itm, prompt_en=pe, log=self.log,
                                                aesthetic=self.aesthetic, ref_images=self.reference_images())
                cr.regen = run_rec
                if run_rec is not None and run_rec.output:
                    flt2 = ag_desc.run(self.agent, [x.path for x in run_rec.output.candidates], sp, pe, kind="candidate", log=self.log)
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
