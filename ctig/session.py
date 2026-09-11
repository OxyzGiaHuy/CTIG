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
    AnalysisResult, CulturalSpec, GenSpec, MultiGenResult, Prompt, QueryComparison, ReviewOutcome, SearchResult,
    from_dict, to_dict,
)


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
        order = ["analysis", "compare", "retrieve", "spec", "genspec", "multigen", "review"]
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
        key = _h({"spec": _h(to_dict(sp)), "pe": a.prompt_en, "t2i": [c.t2i.steps, c.t2i.guidance, c.t2i.width, c.t2i.height,
                                                                          c.t2i.n_candidates, c.t2i.init_negatives], "seed": c.seed})
        return self._memo("genspec", key, GenSpec,
                          lambda: build_initial_spec(self.prompt, sp, a.prompt_en, c.t2i, c.seed, c.t2i.init_negatives), force)

    def multigen(self, models: list[str] | None = None, on_model_done=None, force: bool = False) -> tuple[MultiGenResult, str]:
        from .stages import multigen as st_mg

        models = list(models or self.cfg.models)
        gen, _ = self.genspec()
        sp, _ = self.spec()
        a, _ = self.analysis()
        c = self.cfg
        key = _h({"gen": st_mg.genspec_hash(gen), "models": models, "n": c.multigen.n_candidates, "side": c.multigen.max_side,
                  "ov": c.multigen.overrides})
        lora_dir = Path(c.multigen.lora_dir) if c.multigen.lora_dir else self.cache_dir / "lora"

        def compute():
            return st_mg.run(gen, sp, self.kb, models, c.multigen, self.out_dir, clip=self.clip, itm=self.itm,
                             prompt_en=a.prompt_en or self.prompt.text_en, log=self.log, on_model_done=on_model_done,
                             lora_dir=lora_dir)

        def reusable(v: MultiGenResult) -> bool:
            # hàng "bỏ qua: ..." (only_if_entity) là chủ ý, rẻ, không cần chạy lại; hàng lỗi thật thì phải thử lại
            return bool(v.runs) and all(r.output is not None or (r.error or "").startswith("bỏ qua") for r in v.runs)

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
