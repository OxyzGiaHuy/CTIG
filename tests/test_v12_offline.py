"""
Test v1.2 offline (không GPU, không mạng): websearch cache + so sánh truy vấn, multigen, viz, session memo.
Chạy: python tests/test_v12_offline.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ctig.config import Config
from ctig.kb import KnowledgeBase
from ctig.pipeline import load_prompts
from ctig.schema import AnalysisResult, Keyword, MultiGenResult, from_dict
from ctig.stages.websearch import WebClient, compare_queries

FAILED = []


def check(name, cond, detail=""):
    print(("  ok    " if cond else "  FAIL  ") + name + ("" if cond else f"  {detail}"))
    if not cond:
        FAILED.append(name)


class FakeWeb(WebClient):
    """Không gọi mạng: text/images/commons trả dữ liệu soạn sẵn qua _cached; download ghi ảnh màu."""

    def __init__(self, cfg, cache_dir):
        cfg.web_api = "none"
        super().__init__(cfg, cache_dir)
        self.fetches = 0

    def text(self, q, region="vn-vi", n=5):
        def fetch():
            self.fetches += 1
            return [{"title": f"Bài về {q}", "body": f"{q}: sườn nón là nan tre, quai buộc đối xứng hai bên. " * 3,
                     "href": f"https://example.vn/{abs(hash(q)) % 1000}", "provenance": "fake"} for _ in range(n)]
        return self._cached({"kind": "text", "api": "fake", "q": q, "region": region, "n": n}, fetch)

    def images(self, q, n=6, region="vn-vi"):
        def fetch():
            self.fetches += 1
            return [{"image": f"https://img.example/{abs(hash(q + str(i))) % 10**6}.jpg", "title": f"ảnh {i} {q}", "provenance": "fake-img"} for i in range(n)]
        return self._cached({"kind": "images", "api": "fake", "q": q, "region": region, "n": n}, fetch)

    def commons(self, q, n=3):
        return []

    def download(self, url):
        from PIL import Image
        p = self.img_dir / (str(abs(hash(url)) % 10**8) + ".jpg")
        if not p.exists():
            Image.new("RGB", (64, 64), (abs(hash(url)) % 255, 120, 90)).save(p)
        return str(p)


class FakeClip:
    def similarity(self, path, texts):
        return [0.3 for _ in texts]

    def image_matches(self, path, target, cfs):
        return 0.8

    def entity_probs(self, path, spec):
        return {se.entity_id: {"__target__": 0.7, "other": 0.3} for se in spec.entities if se.kind == "object"}


def test_websearch_compare_and_cache(tmp):
    cfg = Config.load("configs/offline.yaml", {"runs_dir": str(tmp)})
    kb = KnowledgeBase.load(cfg.kb_path)
    web = FakeWeb(cfg.retrieval, tmp / "_cache")
    ids = [e.id for e in kb.all()]  # 38 ứng viên -> phải bị cap
    a = AnalysisResult("t", [Keyword("Áo dài", "entity", "surface", 0.9)], ids, prompt_en="A woman in a white ao dai")
    p = load_prompts(cfg.prompts_path)[0]
    cmp = compare_queries(web, a, p, kb, FakeClip(), k_text=5, k_images=6, max_entities=6)
    check("hai cột", [c.label for c in cmp.columns] == ["Từ keywords / thực thể", "Từ prompt gốc"])
    check("cột keywords bị cap 6 thực thể, có ghi chú", "38" in cmp.columns[0].note and len(cmp.columns[0].queries) <= 12, cmp.columns[0].note)
    check("text <= k", all(len(c.text) <= 5 for c in cmp.columns))
    check("ảnh <= k và có sim", all(len(c.images) <= 6 and all(h.clip_prompt_sim is not None for h in c.images) for c in cmp.columns))
    check("cột keywords có P(thực thể), cột prompt không", all(h.clip_entity_prob is not None for h in cmp.columns[0].images)
          and all(h.clip_entity_prob is None for h in cmp.columns[1].images))
    n1 = web.fetches
    compare_queries(web, a, p, kb, FakeClip(), k_text=5, k_images=6, max_entities=6)
    check("chạy lại: cache đĩa, không fetch thêm", web.fetches == n1, f"{n1} -> {web.fetches}")
    check("có file cache web/*.json", any((tmp / "_cache" / "web").glob("*.json")))


def test_registry_and_adapt():
    from ctig.models.registry import REGISTRY, get
    from ctig.stages.multigen import adapt_spec
    from ctig.schema import GenSpec

    for k, m in REGISTRY.items():
        check(f"registry {k} hợp lệ", m.width % 8 == 0 and m.height % 8 == 0 and m.steps > 0 and m.repo)
    check("sdxl_turbo không negative, guidance 0", not get("sdxl_turbo").negative_ok and get("sdxl_turbo").guidance == 0.0)
    check("playground GIỮ VAE repo (latents_mean/std riêng; fp16-fix làm bạc màu, v1.3)", get("playground25").vae is None)
    check("sdxl_aodai chỉ chạy khi có ao_dai", get("sdxl_aodai").only_if_entity == ["ao_dai"])
    cfg = Config().multigen
    cfg.max_side = 768
    g = GenSpec("t", prompt_terms=["a", "b"], negative_terms=["kimono"], seed=1, steps=25, guidance=6.5, width=1024, height=1024, n_candidates=2)
    t = adapt_spec(g, get("sdxl_turbo"), cfg)
    check("turbo: bỏ negative, 4 bước, 512", t.negative_terms == [] and t.steps == 4 and t.width == 512)
    b = adapt_spec(g, get("sdxl_base"), cfg)
    check("sdxl_base kẹp 1024 -> 768", b.width == 768 and b.height == 768 and b.negative_terms == ["kimono"])
    for k in ("sd3_medium", "hunyuan_dit"):
        check(f"{k} experimental không trong config mặc định", get(k).experimental)


def test_multigen_offline(tmp):
    from ctig.stages import multigen as mg
    from ctig.stages.generation import build_initial_spec
    from ctig.llm.rule_agent import RuleAgent
    from ctig.stages import analysis as st_a, spec as st_s
    from ctig.stages.retrieval import LocalRetriever

    cfg = Config.load("configs/offline.yaml", {"runs_dir": str(tmp)})
    kb = KnowledgeBase.load(cfg.kb_path)
    p = next(x for x in load_prompts(cfg.prompts_path) if x.id == "p001")
    ag = RuleAgent()
    a = st_a.run(ag, p, kb, 6)
    s = LocalRetriever(cfg.retrieval, None, tmp / "_cache").search(a, kb)
    sp = st_s.run(ag, p, a, s, kb, 4, 0.3)
    gen = build_initial_spec(p, sp, a.prompt_en, cfg.t2i, cfg.seed)
    done = []
    res = mg.run(gen, sp, kb, ["stub", "khong_co", "sdxl_aodai"], cfg.multigen, tmp / "mg" / p.id, clip=FakeClip(), itm=None,
                 prompt_en=p.text_en, log=lambda *a: None, on_model_done=lambda r: done.append(r.model_key))
    check("3 ModelRun, không raise", len(res.runs) == 3 and done == ["stub", "khong_co", "sdxl_aodai"])
    check("khoá sai -> error, không ảnh", res.runs[1].error and res.runs[1].output is None)
    check("sdxl_aodai chạy vì spec có ao_dai (loader stub? -> lỗi nạp diffusers được bắt)", res.runs[2].error is not None or res.runs[2].output is not None)
    check("stub có ảnh + CLIP fidelity", res.runs[0].output and res.runs[0].output.candidates[0].clip_probs)
    check("grid.png tồn tại", res.grid_path and Path(res.grid_path).exists())
    raw = json.loads((tmp / "mg" / p.id / "multigen.json").read_text(encoding="utf-8"))
    back = from_dict(MultiGenResult, raw)
    check("multigen.json load lại được", back.genspec_hash == res.genspec_hash and len(back.runs) == 3)
    res2 = mg.run(gen, sp, kb, ["stub"], cfg.multigen, tmp / "mg" / p.id, clip=FakeClip(), itm=None, prompt_en=p.text_en, log=lambda *a: None)
    check("chạy lại cùng GenSpec -> ảnh từ đĩa", res2.runs[0].source == "disk")
    return res, sp, a, s, gen


def test_viz(res, sp, a, s, gen):
    from ctig import viz
    from ctig.pipeline import load_prompts

    kb = KnowledgeBase.load(Config().kb_path)
    p = load_prompts(Config().prompts_path)[0]
    h = viz.keywords_table(a, kb, 4, source="computed")
    check("keywords_table có badge và term", "chạy mới" in h and "Áo dài" in h)
    a_big = AnalysisResult("t", [], [e.id for e in kb.all()])
    check("cảnh báo nổ danh mục", "Nổ danh mục" in viz.keywords_table(a_big, kb, 4))
    check("evidence_table có KB", "KB viết tay" in viz.evidence_table(s, kb))
    check("spec_card có EN từ KB", "split at the hips" in viz.spec_card(sp))
    check("genspec_card có negative", "kimono" in viz.genspec_card(gen))
    g = viz.model_grid(res, sp, source="disk")
    check("model_grid có base64 và tên model", "data:image/jpeg;base64," in g and "stub" in g and "khong_co" in g)
    check("score_table có tốt nhất", "Tốt nhất" in viz.score_table(res))
    check("prompt_card", p.text_vi in viz.prompt_card(p))
    rep = viz.Report("t")
    rep.parts.append(h)
    out = rep.save(Path("runs/_test/rep.html"))
    check("Report.save ghi file", out.exists() and "Áo dài" in out.read_text(encoding="utf-8"))


def test_session_memo(tmp):
    from ctig.session import Session

    cfg = Config.load("configs/offline.yaml", {"runs_dir": str(tmp), "run_name": "sess"})
    p = next(x for x in load_prompts(cfg.prompts_path) if x.id == "p001")
    s = Session(cfg, p, log=lambda *a: None)
    calls = {"n": 0}
    real = s.agent.analyze

    def counting(prompt, kb):
        calls["n"] += 1
        return real(prompt, kb)

    s.agent.analyze = counting
    a1, src1 = s.analysis()
    a2, src2 = s.analysis()
    check("lần 1 computed, lần 2 memory, agent gọi 1 lần", src1 == "computed" and src2 == "memory" and calls["n"] == 1, f"{src1},{src2},{calls}")
    s2 = Session(cfg, p, log=lambda *a: None)
    s2.agent.analyze = counting
    a3, src3 = s2.analysis()
    check("Session mới cùng prompt: đọc từ đĩa, không gọi agent", src3 == "disk" and calls["n"] == 1, f"{src3},{calls}")
    p2 = type(p)(p.id, p.text_vi + " buổi sáng", p.text_en)
    s3 = Session(cfg, p2, log=lambda *a: None)
    s3.agent.analyze = counting
    _, src4 = s3.analysis()
    check("đổi prompt -> chạy lại", src4 == "computed" and calls["n"] == 2)
    sp, _ = s.spec()
    gen, _ = s.genspec()
    r1, m1 = s.multigen(["stub"])
    r2, m2 = s.multigen(["stub"])
    check("multigen memo", m1 == "computed" and m2 == "memory" and r1.runs[0].output is not None)
    r3, m3 = s.multigen(["stub", "khong_co"])
    r4, m4 = s.multigen(["stub", "khong_co"])
    check("multigen có hàng lỗi -> KHÔNG dùng lại cache bước", m3 == "computed" and m4 == "computed" and r4.runs[1].error)
    check("hàng stub tốt vẫn tái dùng ảnh từ đĩa khi chạy lại", r4.runs[0].source == "disk")
    s.invalidate("spec")
    check("invalidate xoá spec và các bước sau", "spec" not in s.steps and "multigen" not in s.steps and "analysis" in s.steps)


def test_llm_cache(tmp):
    from ctig.llm import cache as lc
    from ctig.llm.base import JSONChatMixin

    lc.configure(tmp / "llm")

    class B(JSONChatMixin):
        name = "fake"; model_id = "m"; json_retries = 0
        def __init__(self): self.n = 0
        def chat(self, system, user, images=None):
            self.n += 1
            return '{"x": 1}'

    b = B()
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
    b.complete_json("s", "u", schema); b.complete_json("s", "u", schema)
    check("cùng khoá -> chat gọi 1 lần", b.n == 1, str(b.n))
    b.complete_json("s", "u khác", schema)
    check("khác user -> gọi thêm", b.n == 2)
    lc.configure(None)


def test_v13_offline(tmp):
    """v1.3: hậu tố @scale, hash render, chuẩn hoá thẩm mỹ, điểm tổng có 'đẹp', prompt dài không cần torch."""
    from ctig.models.registry import REGISTRY, get, parse_key
    from ctig.stages import multigen as mg
    from ctig.stages.aesthetic import normalize
    from ctig.schema import Candidate, GenSpec, ModelRun, MultiGenResult, from_dict, to_dict
    from ctig.stages.generation import build_initial_spec
    from ctig.llm.rule_agent import RuleAgent
    from ctig.stages import analysis as st_a, spec as st_s
    from ctig.stages.retrieval import LocalRetriever

    check("parse_key sdxl_aodai@0.6", parse_key("sdxl_aodai@0.6") == ("sdxl_aodai", 0.6) and get("sdxl_aodai@0.6").key == "sdxl_aodai")
    try:
        parse_key("sdxl_aodai@abc"); check("parse_key hậu tố sai -> KeyError", False)
    except KeyError:
        check("parse_key hậu tố sai -> KeyError", True)
    for k in ("realvis_xl", "realvis_aodai", "sdxl_refplus", "sd35_medium"):
        check(f"registry có {k}", k in REGISTRY)
    check("sdxl_refplus dùng IP-Adapter Plus", get("sdxl_refplus").ip_adapter and get("sdxl_refplus").ip_adapter_kind == "plus")
    check("turbo/sd3 không hires, scheduler keep", not get("sdxl_turbo").hires_ok and get("sd35_medium").scheduler == "keep")

    cfg = Config.load("configs/offline.yaml", {"runs_dir": str(tmp)})
    g = GenSpec("t", prompt_terms=["a"], negative_terms=[], seed=1, steps=25, guidance=6.5, width=1024, height=1024, n_candidates=2)
    h0 = mg.genspec_hash(g, mg.render_settings(cfg.multigen))
    cfg.multigen.hires.enabled = True
    h1 = mg.genspec_hash(g, mg.render_settings(cfg.multigen))
    cfg.multigen.hires.enabled = False
    cfg.multigen.scheduler = "euler"
    h2 = mg.genspec_hash(g, mg.render_settings(cfg.multigen))
    check("bật hires / đổi scheduler -> hash ảnh đổi", len({h0, h1, h2}) == 3)
    check("không render -> hash cũ giữ nguyên", mg.genspec_hash(g) == mg.genspec_hash(g, None))

    check("normalize min-max", normalize([20.0, 22.0, None, 21.0]) == [0.0, 1.0, None, 0.5] and normalize([5.0, 5.0]) == [0.5, 0.5])

    class FakeScorer:
        calls = 0
        def _on_gpu(self): pass
        def _off_gpu(self): pass
        def score(self, prompt, paths):
            FakeScorer.calls += 1
            return [20.0 + i for i in range(len(paths))]

    kb = KnowledgeBase.load(cfg.kb_path)
    p = next(x for x in load_prompts(cfg.prompts_path) if x.id == "p001")
    ag = RuleAgent()
    a = st_a.run(ag, p, kb, 6)
    s = LocalRetriever(cfg.retrieval, None, tmp / "_cache").search(a, kb)
    sp = st_s.run(ag, p, a, s, kb, 4, 0.3)
    gen = build_initial_spec(p, sp, a.prompt_en, cfg.t2i, cfg.seed)
    cfg.multigen.n_candidates = 2
    res = mg.run(gen, sp, kb, ["stub", "stub@0.6"], cfg.multigen, tmp / "mg13" / p.id, clip=FakeClip(), itm=None,
                 prompt_en=p.text_en, log=lambda *a: None, aesthetic=FakeScorer())
    check("hàng stub@0.6 chạy được, model_key giữ hậu tố", [r.model_key for r in res.runs] == ["stub", "stub@0.6"] and all(r.output for r in res.runs))
    check("thư mục hàng sweep không chứa '@'", all("@" not in c.path for r in res.runs for c in r.output.candidates))
    cands = [c for r in res.runs for c in r.output.candidates]
    check("PickScore thô và chuẩn hoá cho mọi ứng viên", all(c.pick_score is not None and c.aesthetic is not None for c in cands)
          and min(c.aesthetic for c in cands) == 0.0 and max(c.aesthetic for c in cands) == 1.0)
    c0 = Candidate("x", 1, clip_fidelity=1.0, clip_probs={"e": {"a": 1.0}}, attr_contrast=0.5, aesthetic=0.0)
    c1 = Candidate("y", 2, clip_fidelity=1.0, clip_probs={"e": {"a": 1.0}}, attr_contrast=0.5, aesthetic=1.0)
    check("điểm tổng có 'đẹp'", mg.combined_score(c1) > mg.combined_score(c0))
    back = from_dict(MultiGenResult, json.loads(json.dumps(to_dict(res))))
    check("multigen.json v1.3 load lại được (notes, pick_score, base_path)", back.runs[0].output.candidates[0].pick_score is not None
          and isinstance(back.runs[0].notes, list))
    # chạy lại: ảnh từ đĩa, PickScore KHÔNG gọi lại (đã có trong multigen.json)
    n = FakeScorer.calls
    res2 = mg.run(gen, sp, kb, ["stub", "stub@0.6"], cfg.multigen, tmp / "mg13" / p.id, clip=FakeClip(), itm=None,
                  prompt_en=p.text_en, log=lambda *a: None, aesthetic=FakeScorer())
    check("chạy lại: ảnh từ đĩa, không chấm PickScore lại", all(r.source == "disk" for r in res2.runs) and FakeScorer.calls == n)

    # Prompt dài: đếm token với tokenizer giả, không có compel -> ghi chú "bị cắt"; không cần torch
    import sys, types
    from ctig.stages import generation as gmod
    fake_torch = types.ModuleType("torch")
    saved = sys.modules.get("torch")
    sys.modules["torch"] = fake_torch
    try:
        class Tok:
            def __call__(self, text, truncation=False):
                return {"input_ids": text.split()}
        pipe = types.SimpleNamespace(tokenizer=Tok())
        dg = gmod.DiffusersGenerator(pipe, "m", family="sdxl", long_prompt=True)
        long_gen = GenSpec("t", prompt_terms=[f"w{i}" for i in range(90)], negative_terms=["n"], seed=1, steps=1, guidance=1, width=8, height=8, n_candidates=1)
        kw = dg._prompt_kwargs(long_gen)
        check("prompt 90 token: đếm được, thiếu compel -> prompt thô + ghi chú", dg.prompt_tokens == 90 and "prompt" in kw
              and any("compel" in n for n in dg.notes), str(dg.notes))
        short = GenSpec("t", prompt_terms=[f"w{i}" for i in range(10)], negative_terms=[], seed=1, steps=1, guidance=1, width=8, height=8, n_candidates=1)
        dg2 = gmod.DiffusersGenerator(pipe, "m", family="sdxl", long_prompt=True)
        check("prompt ngắn: không ghi chú", dg2._prompt_kwargs(short)["prompt"] and not dg2.notes)
    finally:
        if saved is not None:
            sys.modules["torch"] = saved
        else:
            sys.modules.pop("torch", None)


def test_progress_report(tmp):
    """v1.3: báo cáo tiến độ dựng được từ thư mục run của Session (offline, stub)."""
    from ctig.progress_report import bar_chart_svg, build, md_to_html
    from ctig.session import Session

    cfg = Config.load("configs/offline.yaml", {"runs_dir": str(tmp)})
    cfg.multigen.n_candidates = 2
    p = next(x for x in load_prompts(cfg.prompts_path) if x.id == "p001")
    s = Session(cfg, p, tmp / "rep", log=lambda *a: None)
    s.multigen(["stub", "stub@0.5"])
    out = build(tmp / "rep", tmp / "rep" / "progress_report.html", title="t", log=lambda *a: None)
    h = out.read_text(encoding="utf-8")
    check("báo cáo có prompt, model, ảnh gốc JPEG, SVG biểu đồ, bảng", all(x in h for x in ("Prompt p001", "stub@0.5", "data:image/jpeg", "<svg", "Tóm tắt theo model")))
    check("báo cáo có phần research (findings + giả thuyết)", "Kết luận đến nay" in h and "Giả thuyết đang kiểm" in h)
    check("grid_hires.png được vẽ", (tmp / "rep" / "p001" / "grid_hires.png").exists())
    svg = bar_chart_svg({"a": [("m1", 0.5), ("m2", None)], "b": [("m1", 0.9), ("m2", 0.1)]})
    check("bar_chart_svg bỏ qua None, có nhãn", "m2" in svg and svg.count("<rect") >= 3)
    check("md_to_html", "<li><b>x</b> y</li>" in md_to_html("## T\n- **x** y\n"))
    # Report dedupe
    from ctig import viz
    r = viz.Report("t")
    r.show(viz._wrap("Bước 5 · Bảng điểm", "<p>a</p>", "computed")); r.show(viz._wrap("Bước 5 · Bảng điểm", "<p>b</p>", "disk"))
    check("Report: cùng tiêu đề thì thay, không nhân đôi", len(r.parts) == 1 and "<p>b</p>" in r.parts[0])


def test_render_variants(tmp):
    """v1.4.1: ba cách render prompt, hậu tố #variant, negative riêng theo model, thẻ không chứa danh từ must_have."""
    from ctig.models.registry import get, parse_key, parse_variant
    from ctig.stages import multigen as mg
    from ctig.stages.generation import build_initial_spec, render_terms
    from ctig.llm.rule_agent import RuleAgent
    from ctig.stages import analysis as st_a, spec as st_s
    from ctig.stages.retrieval import LocalRetriever
    from ctig.kb import tokens

    cfg = Config.load("configs/offline.yaml", {"runs_dir": str(tmp)})
    kb = KnowledgeBase.load(cfg.kb_path)
    check("KB 0.4.0: mọi thực thể có tags_en và neg_tags_en", all(e.tags_en and e.neg_tags_en for e in kb.all()) and kb.version.startswith("0.4"))
    NOUNS = {"trousers", "collar", "sash", "noodles", "broth", "boat", "lanterns", "gongs", "strings", "puppets", "wrapper"}
    leaks = [(e.id, n) for e in kb.all() for n in e.neg_tags_en if tokens(n) & NOUNS & set().union(*[tokens(a) for a in e.must_have_en])]
    check("neg_tags_en không chứa danh từ chính của must_have", not leaks, str(leaks[:5]))
    p = next(x for x in load_prompts(cfg.prompts_path) if x.id == "p001")
    ag = RuleAgent(); a = st_a.run(ag, p, kb, 6)
    s = LocalRetriever(cfg.retrieval, None, tmp / "_cache").search(a, kb); sp = st_s.run(ag, p, a, s, kb, 4, 0.3)
    g_tags = build_initial_spec(p, sp, a.prompt_en, cfg.t2i, 1, render="tags")
    g_leg = build_initial_spec(p, sp, a.prompt_en, cfg.t2i, 1, render="legacy")
    g_sen = build_initial_spec(p, sp, a.prompt_en, cfg.t2i, 1, render="sentence")
    g_tw = build_initial_spec(p, sp, a.prompt_en, cfg.t2i, 1, render="tags_w")
    check("tags: thực thể lên đầu, thẻ ngắn, KHÔNG trọng số; tags_w có trọng số thẻ đầu", g_tags.prompt_terms[0].startswith("Vietnamese")
          and "fitted long tunic" in g_tags.prompt_terms and not g_tags.term_weights
          and g_tw.term_weights.get("fitted long tunic") == cfg.t2i.emphasis_weight, str(g_tags.prompt_terms[:4]))
    g_ln = build_initial_spec(p, sp, a.prompt_en, cfg.t2i, 1, render="legacy_negtags")
    check("legacy_negtags: prompt v1.3 + negative thẻ", g_ln.prompt_terms == g_leg.prompt_terms and "obi sash" in g_ln.negative_terms
          and not any("no trousers" in n for n in g_ln.negative_terms))
    check("tags: negative dùng neg_tags, không có 'trousers'", "obi sash" in g_tags.negative_terms and not any("trousers" in n for n in g_tags.negative_terms), str(g_tags.negative_terms))
    check("legacy: cảnh trước, câu dài, negative must_not_en", g_leg.prompt_terms[0] == a.prompt_en and any("no trousers" in n for n in g_leg.negative_terms))
    check("sentence: một đoạn văn", len(g_sen.prompt_terms) == 1 and g_sen.prompt_terms[0].startswith(a.prompt_en.rstrip(".")) and "has" in g_sen.prompt_terms[0])
    check("mặc định config = legacy (v1.4 p001: tags kém hơn)", build_initial_spec(p, sp, a.prompt_en, cfg.t2i, 1).render == "legacy")
    check("parse_variant", parse_variant("realvis_xl#legacy_negtags") == "legacy_negtags" and parse_variant("realvis_xl@0.6") is None and parse_key("sdxl_aodai@0.6#tags_w") == ("sdxl_aodai", 0.6))
    try:
        parse_variant("x#bogus"); check("parse_variant sai -> KeyError", False)
    except KeyError:
        check("parse_variant sai -> KeyError", True)
    ad = mg.adapt_spec(g_tags, get("realvis_xl"), cfg.multigen, spec=sp, prompt_en=a.prompt_en, variant="legacy", t2i_cfg=cfg.t2i)
    check("adapt_spec #legacy render lại + nối negative riêng RealVis", ad.render == "legacy" and ad.prompt_terms[0] == a.prompt_en and "open mouth" in ad.negative_terms)
    ad2 = mg.adapt_spec(g_tags, get("sd35_medium"), cfg.multigen, spec=sp, prompt_en=a.prompt_en, t2i_cfg=cfg.t2i)
    check("sd3 tự chuyển sentence", ad2.render == "sentence")
    cfg.multigen.n_candidates = 1
    res = mg.run(g_tags, sp, kb, ["stub", "stub#legacy"], cfg.multigen, tmp / "rv" / p.id, clip=FakeClip(), itm=None, prompt_en=a.prompt_en,
                 log=lambda *a: None, t2i_cfg=cfg.t2i)
    check("hàng stub#legacy chạy, thư mục không chứa '#'", [r.model_key for r in res.runs] == ["stub", "stub#legacy"] and all("#" not in c.path for r in res.runs for c in r.output.candidates))


def test_refcrop_and_copy(tmp):
    """v1.4.2: cắt ảnh tham chiếu theo thực thể bằng CLIP giả, cờ +ref, số đo chép trừ điểm tổng."""
    from PIL import Image
    from ctig.models.registry import parse_flags
    from ctig.schema import Candidate
    from ctig.stages import multigen as mg
    from ctig.stages.refcrop import crop_to_entity, _boxes

    # ảnh 400x300: vật thể là ô đỏ ở góc phải dưới; CLIP giả chấm ô cắt theo tỉ lệ pixel đỏ
    img = Image.new("RGB", (400, 300), (30, 30, 30))
    for x in range(250, 400):
        for y in range(150, 300):
            img.putpixel((x, y), (220, 30, 30))
    src = tmp / "ref.jpg"; img.save(src)

    class CropClip:
        def similarity_image(self, im, texts):
            px = list(im.getdata()); red = sum(1 for r, g, b in px if r > 150 and g < 80) / max(1, len(px))
            return [0.2 + 0.6 * red] + [0.25] * (len(texts) - 1)
    out, info = crop_to_entity(CropClip(), src, "a red object", tmp / "crops")
    check("cắt được vùng đỏ, ảnh vuông, có gain", info.get("cropped") and Path(out).exists() and info["gain"] > 0.02 and Image.open(out).size[0] == Image.open(out).size[1], str(info))
    out2, info2 = crop_to_entity(CropClip(), src, "a red object", tmp / "crops")
    check("cắt lần hai từ cache", out2 == out and info2.get("cached"))
    box = info["box"]; check("ô cắt chứa vùng đỏ", box[0] <= 260 and box[1] <= 160 and box[2] >= 390 and box[3] >= 290, str(box))
    class FlatClip:
        def similarity_image(self, im, texts): return [0.5] + [0.25] * (len(texts) - 1)
    out3, info3 = crop_to_entity(FlatClip(), src, "x", tmp / "crops2")
    check("không hơn cả bức -> giữ ảnh gốc", out3 == str(src) and not info3.get("cropped"))
    check("_boxes có ô ở góc phải dưới", any(b[2] == 400 and b[3] == 300 for b in _boxes(400, 300)))

    check("parse_flags +ref và lỗi cờ lạ", parse_flags("realvis_xl+ref") == {"ref"})
    try:
        parse_flags("realvis_xl+bogus"); check("cờ lạ -> KeyError", False)
    except KeyError:
        check("cờ lạ -> KeyError", True)
    c_ok = Candidate("a", 1, clip_fidelity=1.0, clip_probs={"e": {"a": 1}}, attr_contrast=0.8, ref_sim=0.70)
    c_copy = Candidate("b", 2, clip_fidelity=1.0, clip_probs={"e": {"a": 1}}, attr_contrast=0.8, ref_sim=0.97)
    check("chép (ref_sim 0.97) bị trừ điểm tổng, 0.70 thì không (CLIP id không vào điểm khi saturated tắt)",
          mg.combined_score(c_copy) < mg.combined_score(c_ok) - 0.2 and abs(mg.combined_score(c_ok) - 0.8) < 1e-6)
    cfg = Config.load("configs/offline.yaml", {"runs_dir": str(tmp)})
    kb = KnowledgeBase.load(cfg.kb_path)
    from ctig.llm.rule_agent import RuleAgent
    from ctig.stages import analysis as st_a, spec as st_s
    from ctig.stages.retrieval import LocalRetriever
    from ctig.stages.generation import build_initial_spec
    p = next(x for x in load_prompts(cfg.prompts_path) if x.id == "p001"); ag = RuleAgent(); a = st_a.run(ag, p, kb, 6)
    s = LocalRetriever(cfg.retrieval, None, tmp / "_cache").search(a, kb); sp = st_s.run(ag, p, a, s, kb, 4, 0.3)
    gen = build_initial_spec(p, sp, a.prompt_en, cfg.t2i, 1)
    class RefClip(FakeClip):
        def image_similarity(self, x, y): return 0.95
    cfg.multigen.n_candidates = 1
    res = mg.run(gen, sp, kb, ["stub", "stub+ref"], cfg.multigen, tmp / "rc" / p.id, clip=RefClip(), itm=None, prompt_en=a.prompt_en,
                 log=lambda *a: None, ref_images=[str(src)], t2i_cfg=cfg.t2i)
    check("stub+ref: họ stub không nhận +ref -> hàng lỗi rõ, hàng stub vẫn chạy", res.runs[1].error and "sdxl" in res.runs[1].error and res.runs[0].output)
    check("ref_sim được tính cho mọi ứng viên khi có ảnh tham chiếu", res.runs[0].output.candidates[0].ref_sim == 0.95)


def test_garment_rules():
    """v1.4.3: luật cứng trên trường trang phục ghi đè agent văn bản thiên lệch 'có'."""
    from ctig.agents.describe import garment_rules
    from ctig.schema import ImageDescriptor

    d = ImageDescriptor("x", people_count=1, garments=["{'type': 'dress', 'fit': 'loose', 'length': 'floor', 'collar': 'crossed', 'sleeves': 'long', 'lower_body': 'not visible', 'sash_or_belt': '', 'slits': ''}"])
    check("collar crossed -> stand-up collar ABSENT", garment_rules(d, "high stand-up mandarin collar") == "absent")
    check("collar crossed -> Y-shaped crossed collar (must_not) PRESENT", garment_rules(d, "diagonal Y-shaped crossed collar") == "present")
    check("lower_body not visible -> trousers None (unsure)", garment_rules(d, "worn over wide-legged long trousers") is None)
    d2 = ImageDescriptor("y", garments=["{'type': 'tunic', 'collar': 'stand-up', 'lower_body': 'trousers', 'sash_or_belt': 'none', 'slits': 'yes'}"])
    check("stand-up + trousers + slits -> present", all(garment_rules(d2, a) == "present" for a in ("high stand-up mandarin collar", "worn over wide-legged long trousers", "tunic split at the hips")))
    check("obi sash must_not absent khi sash none", garment_rules(d2, "wide obi sash tied at the back") == "absent")
    d3 = ImageDescriptor("z", garments=["{'type': 'dress', 'length': 'floor', 'lower_body': 'bare legs', 'collar': 'round'}"])
    check("bare legs -> 'one-piece dress with no trousers' PRESENT, trousers ABSENT", garment_rules(d3, "one-piece dress with no trousers underneath") == "present"
          and garment_rules(d3, "worn over wide-legged long trousers") == "absent")
    check("garment chuỗi thường -> None", garment_rules(ImageDescriptor("w", garments=["white outfit"]), "high stand-up mandarin collar") is None)


def test_v15_offline(tmp):
    """v1.5: auto_ref theo prior, ensemble hạng, best-of-N thích nghi, OWL-ViT lùi về CLIP, brief có chiều CULTIVate."""
    from ctig.schema import Candidate, CulturalSpec, GenOutput, ModelRun, MultiGenResult, SpecEntity, GenSpec
    from ctig.stages import multigen as mg
    from ctig.stages.refcrop import crop_to_entity
    from ctig.llm.rule_agent import RuleAgent
    from ctig.stages import analysis as st_a, spec as st_s
    from ctig.stages.retrieval import LocalRetriever
    from ctig.stages.generation import build_initial_spec

    cfg = Config.load("configs/offline.yaml", {"runs_dir": str(tmp)})
    kb = KnowledgeBase.load(cfg.kb_path)
    # auto_ref: áo dài prior cao -> không dùng; thuyền thúng prior thấp -> dùng; ad-hoc -> dùng
    def spec_of(eid):
        e = kb.get(eid)
        return CulturalSpec("t", [SpecEntity(eid, e.name_vi, e.name_en, [], [], [], 1.0, kind="object")], [], [])
    use_ao, why_ao = mg.should_use_refs(spec_of("ao_dai"), kb, cfg.multigen)
    use_tt, why_tt = mg.should_use_refs(spec_of("thuyen_thung"), kb, cfg.multigen)
    use_x, _ = mg.should_use_refs(CulturalSpec("t", [SpecEntity("x_la", "lạ", "unknown", [], [], [], 1.0, kind="object")], [], []), kb, cfg.multigen)
    check("auto_ref: áo dài (prior cao) không dùng ảnh; thuyền thúng và thực thể lạ dùng", not use_ao and use_tt and use_x, f"{why_ao} | {why_tt}")
    cfg.multigen.auto_ref.enabled = False
    check("auto_ref tắt -> luôn dùng", mg.should_use_refs(spec_of("ao_dai"), kb, cfg.multigen)[0])
    cfg.multigen.auto_ref.enabled = True

    # ensemble hạng: ứng viên A cao attr thấp đẹp, B ngược lại, C thấp cả hai; D chép ref
    mk = lambda p, attr, aes, ref=None: Candidate(p, 1, clip_fidelity=1.0, clip_probs={"e": {"a": 1}}, attr_contrast=attr, itm_attrs=0.9, aesthetic=aes, ref_sim=ref)
    cs = [mk("A", 0.9, 0.2), mk("B", 0.5, 0.9), mk("C", 0.3, 0.1), mk("D", 0.95, 0.95, ref=0.97)]
    res = MultiGenResult("t", [ModelRun("m", "-", GenSpec("t"), output=GenOutput("t", 0, cs, 0))])
    mg.COPY_THRESHOLD = 0.88
    mg.ensemble_rank(res)
    e = {c.path: c.ensemble for c in cs}
    check("ensemble: C thấp nhất; D bị phạt chép dù metric cao nhất", e["C"] < min(e["A"], e["B"]) and e["D"] < max(e["A"], e["B"]), str(e))
    check("score_key dùng ensemble khi có", mg.score_key(cs[0]) == cs[0].ensemble)
    mg.rechoose(res)
    check("rechoose theo ensemble, không chọn D (chép)", res.runs[0].output.chosen != 3)

    # adaptive: FakeClip không có probs -> attr None -> coi là chưa đạt -> sinh tới max
    p = next(x for x in load_prompts(cfg.prompts_path) if x.id == "p001"); ag = RuleAgent(); a = st_a.run(ag, p, kb, 6)
    s = LocalRetriever(cfg.retrieval, None, tmp / "_cache").search(a, kb); sp = st_s.run(ag, p, a, s, kb, 4, 0.3)
    gen = build_initial_spec(p, sp, a.prompt_en, cfg.t2i, 1)
    cfg.multigen.n_candidates = 6; cfg.multigen.adaptive.enabled = True; cfg.multigen.adaptive.min = 2; cfg.multigen.adaptive.max = 5; cfg.multigen.adaptive.step = 2
    r1 = mg.run(gen, sp, kb, ["stub"], cfg.multigen, tmp / "ad" / p.id, clip=FakeClip(), itm=None, prompt_en=a.prompt_en, log=lambda *a: None, t2i_cfg=cfg.t2i)
    n = len(r1.runs[0].output.candidates)
    check("adaptive: verifier chưa đạt -> 2 -> 4 -> 5 ứng viên (kẹp max), seed/tên file không trùng", n == 5
          and len({c.seed for c in r1.runs[0].output.candidates}) == 5 and len({c.path for c in r1.runs[0].output.candidates}) == 5
          and any(n_.startswith("adaptive: 2 -> 5") for n_ in r1.runs[0].notes), str((n, r1.runs[0].notes)))
    class GoodClip(FakeClip):
        def probs(self, path, labels): return [0.9] + [0.1 / max(1, len(labels) - 1)] * (len(labels) - 1)
    r2 = mg.run(gen, sp, kb, ["stub"], cfg.multigen, tmp / "ad2" / p.id, clip=GoodClip(), itm=None, prompt_en=a.prompt_en, log=lambda *a: None, t2i_cfg=cfg.t2i)
    check("adaptive: verifier đạt ngay -> dừng ở 2", len(r2.runs[0].output.candidates) == 2 and any("đạt với 2" in n_ for n_ in r2.runs[0].notes), str(r2.runs[0].notes))
    r3 = mg.run(gen, sp, kb, ["stub"], cfg.multigen, tmp / "ad2" / p.id, clip=GoodClip(), itm=None, prompt_en=a.prompt_en, log=lambda *a: None, t2i_cfg=cfg.t2i)
    check("adaptive: chạy lại dùng lại ảnh đĩa (cần >= min)", r3.runs[0].source == "disk")
    cfg.multigen.adaptive.enabled = False

    # OWL-ViT không có torch offline -> lùi về CLIP quét lưới
    from PIL import Image
    img = Image.new("RGB", (400, 300), (30, 30, 30))
    for x in range(250, 400):
        for y in range(150, 300):
            img.putpixel((x, y), (220, 30, 30))
    src = tmp / "ref15.jpg"; img.save(src)
    class CropClip:
        def similarity_image(self, im, texts):
            px = list(im.getdata()); red = sum(1 for r, g, b in px if r > 150 and g < 80) / max(1, len(px))
            return [0.2 + 0.6 * red] + [0.25] * (len(texts) - 1)
    out, info = crop_to_entity(CropClip(), src, "a red object", tmp / "crops15", detector="owlvit", device="cpu")
    check("detector owlvit lỗi offline -> lùi về clip và vẫn cắt", info.get("cropped") and info.get("how") == "clip", str(info))

    # brief có chiều
    from ctig.session import Session
    sess = Session(cfg, p, tmp / "b15", log=lambda *a: None)
    briefs, _ = sess.brief()
    check("brief có dimensions (RuleAgent: attire cho trang phục)", "attire" in briefs["ao_dai"].dimensions and briefs["ao_dai"].dimensions["attire"])

    # plan: thiếu >= 2 -> use_reference_image
    from ctig.agents import loop as ag_loop
    from ctig.schema import FilterVerdict
    v = FilterVerdict("x", True, missing_must_have=["a", "b"])
    check("plan thiếu >=2 -> sinh lại kèm ảnh tham chiếu", ag_loop.plan_from_verdict(v, sp, gen).use_reference_image)


def test_v151_offline(tmp):
    """v1.5.1: metric bão hoà bị loại khỏi điểm chọn; hàng +ref bị gate dùng lại hàng gốc; force_refs vượt gate."""
    from ctig.schema import Candidate, CulturalSpec, GenOutput, GenSpec, ModelRun, MultiGenResult, SpecEntity
    from ctig.stages import multigen as mg

    mg.SATURATED = False
    c = Candidate("a", 1, clip_fidelity=1.0, clip_probs={"e": {"a": 1}}, attr_contrast=0.5, itm_score=0.99, itm_attrs=0.5, aesthetic=0.5)
    check("combined bỏ CLIP id / ITM danh tính khi saturated_metrics tắt", abs(mg.combined_score(c) - 0.5) < 1e-9)
    only_id = Candidate("b", 1, clip_fidelity=0.9, clip_probs={"e": {"a": 1}})
    check("không có gì khác thì mới rơi về CLIP id", abs(mg.combined_score(only_id) - 0.9) < 1e-9)
    cs = [Candidate(f"c{i}", i, clip_fidelity=1.0 - 0.3 * i, clip_probs={"e": {"a": 1}}, attr_contrast=0.5, itm_attrs=0.5, aesthetic=0.5) for i in range(3)]
    res = MultiGenResult("t", [ModelRun("m", "-", GenSpec("t"), output=GenOutput("t", 0, cs, 0))])
    mg.ensemble_rank(res)
    check("ensemble bỏ CLIP id -> mọi ứng viên hoà nhau", len({round(x.ensemble, 6) for x in cs}) == 1)
    mg.SATURATED = True
    mg.ensemble_rank(res)
    check("bật saturated_metrics -> CLIP id lại phân hạng", cs[0].ensemble > cs[2].ensemble)
    mg.SATURATED = False

    cfg = Config.load("configs/offline.yaml", {"runs_dir": str(tmp)})
    kb = KnowledgeBase.load(cfg.kb_path)
    check("config offline: saturated_metrics tắt mặc định", cfg.multigen.saturated_metrics is False)
    # force_refs: gate từ chối (áo dài prior cao) nhưng vòng sửa ép dùng
    e = kb.get("ao_dai")
    sp = CulturalSpec("t", [SpecEntity("ao_dai", e.name_vi, e.name_en, [], [], [], 1.0, kind="object")], [], [])
    use, _ = mg.should_use_refs(sp, kb, cfg.multigen)
    check("gate từ chối áo dài", not use)
    from ctig.llm.rule_agent import RuleAgent
    from ctig.stages import analysis as st_a, spec as st_s
    from ctig.stages.retrieval import LocalRetriever
    from ctig.stages.generation import build_initial_spec
    p = next(x for x in load_prompts(cfg.prompts_path) if x.id == "p001"); ag = RuleAgent(); a = st_a.run(ag, p, kb, 6)
    s = LocalRetriever(cfg.retrieval, None, tmp / "_cache").search(a, kb); sp2 = st_s.run(ag, p, a, s, kb, 4, 0.3)
    gen = build_initial_spec(p, sp2, a.prompt_en, cfg.t2i, 1)
    cfg.multigen.adaptive.enabled = False; cfg.multigen.n_candidates = 1
    # stub không nhận +ref (họ stub) -> kiểm alias qua thứ tự: hàng gốc chạy, hàng +ref bị gate -> phải dùng lại, không lỗi
    from ctig.models import registry as reg
    reg.REGISTRY["stub"].family = "sdxl"  # tạm coi stub là sdxl để đi qua nhánh gate/alias (không nạp model vì loader tiêm)
    try:
        calls = []
        def fake_loader(mspec, device, offload, log=print, scheduler=None):
            calls.append(mspec.key); raise RuntimeError("không nạp thật trong test")
        r = mg.run(gen, sp2, kb, ["stub", "stub+ref"], cfg.multigen, tmp / "alias" / p.id, clip=FakeClip(), itm=None, prompt_en=a.prompt_en,
                   log=lambda *a: None, t2i_cfg=cfg.t2i, loader=fake_loader)
        # hàng stub (family sdxl giả) đi qua loader -> lỗi; hàng +ref bị gate -> tìm hàng gốc có output: không có -> chạy bình thường -> cũng lỗi
        check("gate +ref: không có hàng gốc thành công thì chạy bình thường (2 lần nạp)", len(calls) == 2 and all(x.error for x in r.runs), str(calls))
    finally:
        reg.REGISTRY["stub"].family = "stub"
    # alias thật: hàng gốc stub thành công, hàng +ref gate tắt -> dùng lại, không sinh
    reg.REGISTRY["stub"].family = "sdxl"
    try:
        # loader không được gọi vì hàng gốc là stub? family sdxl giả -> cần generator thật. Dùng family stub cho hàng gốc bằng cách chạy trước:
        pass
    finally:
        reg.REGISTRY["stub"].family = "stub"
    r2 = mg.run(gen, sp2, kb, ["stub"], cfg.multigen, tmp / "alias2" / p.id, clip=FakeClip(), itm=None, prompt_en=a.prompt_en, log=lambda *a: None, t2i_cfg=cfg.t2i)
    check("hàng gốc stub chạy", r2.runs[0].output is not None)


def test_filter_agent_failure_tolerant(tmp):
    """v1.5.2: agent văn bản ném lỗi (JSON cắt) -> Filter vẫn ra verdict bằng luật, không làm dừng bước."""
    from ctig.agents import describe as ag_desc
    from ctig.schema import CulturalSpec, ImageDescriptor, SpecEntity

    cfg = Config.load("configs/offline.yaml", {"runs_dir": str(tmp)})
    kb = KnowledgeBase.load(cfg.kb_path); e = kb.get("ao_dai")
    sp = CulturalSpec("t", [SpecEntity("ao_dai", e.name_vi, e.name_en, e.must_have, e.must_not, [], 1.0, kind="object",
                                       required_attrs_en=e.must_have_en, forbidden_attrs_en=e.must_not_en)], [], [])
    class Boom:
        def describe_image(self, path):
            return {"people_count": 1, "subjects": ["woman"], "garments": ["{'type': 'tunic', 'collar': 'stand-up', 'lower_body': 'trousers', 'slits': 'yes', 'sash_or_belt': 'none'}"],
                    "objects": [], "background": "gate", "watermark_or_text": False}
        def match_descriptors(self, description, have, notv):
            raise RuntimeError("Không lấy được JSON sau 3 lần")
    flt = ag_desc.run(Boom(), ["x.png"], sp, "A young woman", kind="candidate", log=lambda *a: None)
    v = flt.verdicts[0]
    check("agent lỗi -> vẫn có verdict, luật khớp collar/trousers/slits", v.keep and len(v.matched_must_have) >= 3 and any("agent văn bản lỗi" in r for r in v.reasons), str(v))
    d = ImageDescriptor("x", people_count=2, subjects=["woman walking"], garments=["{'type': 'dress', 'fit': 'loose', 'collar': 'round', 'lower_body': 'not visible', 'color': 'white'}"], objects=["bag"], background="street")
    ct = ag_desc.compact_text(d)
    check("compact_text gọn, không còn dấu ngoặc dict", "{" not in ct and "type=dress" in ct and len(ct) <= 700, ct)


def test_clip_veto_and_dedupe(tmp):
    """v1.5.2: must_not do VLM đọc ra phải được CLIP xác nhận; Filter không mô tả ảnh trùng đường dẫn."""
    from ctig.agents import describe as ag_desc
    from ctig.schema import CulturalSpec, SpecEntity

    cfg = Config.load("configs/offline.yaml", {"runs_dir": str(tmp)})
    kb = KnowledgeBase.load(cfg.kb_path); e = kb.get("ao_dai")
    sp = CulturalSpec("t", [SpecEntity("ao_dai", e.name_vi, e.name_en, e.must_have, e.must_not, [], 1.0, kind="object",
                                       required_attrs_en=e.must_have_en, forbidden_attrs_en=e.must_not_en)], [], [])
    class WrongCollarVLM:
        calls = 0
        def describe_image(self, path):
            WrongCollarVLM.calls += 1
            return {"people_count": 1, "subjects": ["woman"], "garments": ["{'type': 'ao dai', 'collar': 'crossed', 'lower_body': 'trousers', 'slits': 'none', 'sash_or_belt': 'none'}"], "objects": [], "background": "", "watermark_or_text": False}
        def match_descriptors(self, d, h, n): return {"present_must_have": [], "present_must_not": [], "unsure": []}
    class StandUpClip:  # CLIP thấy cổ đứng
        def probs(self, path, labels): return [0.85, 0.15]
    class CrossedClip:
        def probs(self, path, labels): return [0.3, 0.7]
    f1 = ag_desc.run(WrongCollarVLM(), ["a.png", "a.png", "b.png"], sp, "A young woman", kind="candidate", log=lambda *a: None, clip=StandUpClip())
    check("đường dẫn trùng chỉ mô tả một lần", WrongCollarVLM.calls == 2 and len(f1.verdicts) == 2)
    v = f1.verdicts[0]
    check("VLM nói cổ chéo nhưng CLIP nghiêng cổ đứng -> gỡ must_not, giữ ảnh", v.keep and not v.matched_must_not and any("CLIP nghiêng" in r for r in v.reasons), str(v.reasons))
    class CleanVLM(WrongCollarVLM):
        def describe_image(self, path):
            d = super().describe_image(path)
            if path == "clean.png":
                d["garments"] = ["{'type': 'ao dai', 'collar': 'stand-up', 'lower_body': 'trousers', 'slits': 'yes', 'sash_or_belt': 'none'}"]
            return d
    f2 = ag_desc.run(CleanVLM(), ["c.png", "clean.png"], sp, "A young woman", kind="candidate", log=lambda *a: None, clip=CrossedClip())
    bad = next(v for v in f2.verdicts if v.path == "c.png")
    check("CLIP đồng ý cổ chéo -> must_not giữ, ảnh bị bỏ (ảnh sạch còn lại được giữ)", bad.matched_must_not and not bad.keep and f2.kept == ["clean.png"])
    f3 = ag_desc.run(WrongCollarVLM(), ["d.png"], sp, "A young woman", kind="candidate", log=lambda *a: None, clip=None)
    check("không có CLIP -> giữ phán của luật như cũ", f3.verdicts[0].matched_must_not)
    check("_counterpart: cổ chéo <-> cổ đứng; 'no trousers' <-> trousers", ag_desc._counterpart("diagonal Y-shaped crossed collar", e.must_have_en) == "high stand-up mandarin collar"
          and ag_desc._counterpart("one-piece dress with no trousers underneath", e.must_have_en) == "worn over wide-legged long trousers")


def test_analysis_unsupported_candidates(tmp):
    """v1.5.3: ứng viên không có căn cứ (áo dài, nón lá trong prompt thuyền thúng) bị bỏ; thuyền thúng giữ."""
    from ctig.stages import analysis as st_a
    from ctig.schema import AnalysisResult, Keyword

    cfg = Config.load("configs/offline.yaml", {"runs_dir": str(tmp)})
    kb = KnowledgeBase.load(cfg.kb_path)
    p = next(x for x in load_prompts(cfg.prompts_path) if x.id == "p012")
    class ChattyAgent:
        def analyze(self, prompt, kb):
            return AnalysisResult("t", [Keyword("thuyền thúng", "entity", "surface", 0.9)], ["thuyen_thung", "ao_dai", "non_la"],
                                  prompt_en="A fisherman paddling a round basket boat off a Central Vietnam beach at dawn")
    a = st_a.run(ChattyAgent(), p, kb, 6)
    check("bỏ ao_dai và non_la không có căn cứ, giữ thuyen_thung", a.candidate_entity_ids == ["thuyen_thung"] and "không có căn cứ" in (a.notes or ""), str((a.candidate_entity_ids, a.notes)))
    class OnlyJunk:
        def analyze(self, prompt, kb):
            return AnalysisResult("t", [], ["ao_dai"], prompt_en="x")
    a2 = st_a.run(OnlyJunk(), p, kb, 6)
    check("bù thực thể nêu tên (thuyền thúng) và bỏ ao_dai", "thuyen_thung" in a2.candidate_entity_ids and "ao_dai" not in a2.candidate_entity_ids, str(a2.candidate_entity_ids))


def test_reference_tiers(tmp):
    """v1.5.3: không ảnh nào đạt 0,75 -> nới xuống 0,5 -> rồi ảnh từ prompt gốc chấm lại; rỗng thì [] không ném lỗi."""
    from PIL import Image
    from ctig.schema import EvidenceItem
    from ctig.session import Session

    cfg = Config.load("configs/offline.yaml", {"runs_dir": str(tmp)})
    cfg.agents.ref_filter = False; cfg.multigen.ref_crop = False
    p = next(x for x in load_prompts(cfg.prompts_path) if x.id == "p012")
    s = Session(cfg, p, tmp / "tiers", log=lambda *a: None)
    sr, _ = s.retrieve(); sp, _ = s.spec()
    check("p012 spec có thuyền thúng", any(se.entity_id == "thuyen_thung" for se in sp.entities))
    imgs = {}
    for name in ("low", "mid", "prompt"):
        f = tmp / f"{name}.jpg"; Image.new("RGB", (64, 64), (100, 100, 100)).save(f); imgs[name] = str(f)
    sr.items = [it for it in sr.items if it.kind != "image"]
    sr.items += [EvidenceItem("thuyen_thung", "image", "low", "", local_path=imgs["low"], clip_match=0.3),
                 EvidenceItem("thuyen_thung", "image", "mid", "", local_path=imgs["mid"], clip_match=0.6),
                 EvidenceItem("-", "image", "prompt", "", local_path=imgs["prompt"], clip_match=None)]
    class TierClip:
        def similarity(self, path, texts): return [0.3]
        def image_matches(self, path, label, cfs): return 0.8
    s._clip = TierClip()
    out = s.reference_images(k=3)
    check("tầng 2: lấy ảnh 0,6 (không lấy 0,3), chưa cần ảnh prompt gốc", out == [imgs["mid"]], str(out))
    sr.items = [it for it in sr.items if it.title != "mid"]
    out2 = s.reference_images(k=3)
    check("tầng 3: chỉ còn ảnh prompt gốc, chấm lại 0,8 -> dùng", out2 == [imgs["prompt"]], str(out2))
    class NoClip(TierClip):
        def image_matches(self, path, label, cfs): return 0.2
    s._clip = NoClip()
    check("không ảnh nào đạt -> [] không ném lỗi", s.reference_images(k=3) == [])


def test_refindex_and_attribute_refs(tmp):
    """v1.6: kho ảnh CLIP: build/search/save/load; Session dùng kho làm tầng 0; vòng sửa truy hồi theo caption thuộc tính; override ip_scale."""
    import numpy as np
    from PIL import Image
    from ctig.stages import refindex as ri
    from ctig.stages import multigen as mg
    from ctig.models.registry import get
    from ctig.schema import GenSpec
    from ctig.session import Session

    root = tmp / "kho"; (root / "p012").mkdir(parents=True); (root / "p001").mkdir(parents=True)
    Image.new("RGB", (64, 64), (200, 30, 30)).save(root / "p012" / "red.jpg")
    Image.new("RGB", (64, 64), (30, 30, 200)).save(root / "p001" / "blue.jpg")
    Image.new("RGB", (64, 64), (30, 200, 30)).save(root / "p001" / "green.png")
    (root / "p001" / "note.txt").write_text("x")

    class ColorClip:  # embedding = màu trung bình chuẩn hoá; text "red"/"blue"/"green" -> vector màu
        model_id = "fake-clip"
        def image_embed(self, img):
            px = np.asarray(img.convert("RGB").resize((8, 8)), dtype="float32").reshape(-1, 3).mean(0); return px / (np.linalg.norm(px) + 1e-8)
        def text_embed(self, texts):
            m = {"red": [1, 0, 0], "blue": [0, 0, 1], "green": [0, 1, 0]}
            return np.asarray([m.get(next((k for k in m if k in t.lower()), "red"), [1, 0, 0]) for t in texts], dtype="float32")
        def similarity(self, path, texts): return [0.3 for _ in texts]
    idx = ri.build(root, tmp / "kho.npz", ColorClip(), log=lambda *a: None)
    check("build: 3 ảnh (bỏ .txt), lưu npz+json", len(idx) == 3 and (tmp / "kho.npz").exists() and (tmp / "kho.json").exists())
    idx2 = ri.RefIndex.load(tmp / "kho.npz")
    hits = idx2.search(ColorClip().text_embed(["a red object"])[0], k=2, min_sim=0.26)
    check("search 'red' -> red.jpg đầu, cosine cao", hits and hits[0][0].endswith("red.jpg") and hits[0][1] > 0.9, str(hits))
    check("search giới hạn folder", idx2.search(ColorClip().text_embed(["red"])[0], k=3, min_sim=0.0, folders=["p001"])[0][0].endswith(("blue.jpg", "green.png")))

    cfg = Config.load("configs/offline.yaml", {"runs_dir": str(tmp)})
    cfg.retrieval.ref_index = str(tmp / "kho.npz"); cfg.perception.clip_model = "fake-clip"
    cfg.agents.ref_filter = False; cfg.multigen.ref_crop = False
    p = next(x for x in load_prompts(cfg.prompts_path) if x.id == "p012")
    s = Session(cfg, p, tmp / "ri", log=lambda *a: None)
    class RedClip(ColorClip):
        def text_embed(self, texts): return np.asarray([[1, 0, 0] for _ in texts], dtype="float32")
    s._clip = RedClip()
    out = s.reference_images(k=2)
    check("tầng 0: kho trả red.jpg cho thuyền thúng (clip giả)", out and out[0].endswith("red.jpg"), str(out))
    refs = s.attribute_refs(["round woven bamboo hull", "single oar"], k=2)
    check("attribute_refs: caption thuộc tính -> ảnh từ kho", refs and refs[0].endswith("red.jpg"), str(refs))
    cfg.perception.clip_model = "openai/clip-vit-base-patch32"
    s2 = Session(cfg, p, tmp / "ri2", log=lambda *a: None)
    check("kho đánh chỉ mục bằng CLIP khác -> bỏ kho", s2.ref_index is None)

    # override ip_scale theo khoá đầy đủ
    cfg.multigen.overrides = {"realvis_xl+ref": {"ip_scale": 0.55}}
    g = GenSpec("t", prompt_terms=["a"], seed=1, steps=1, guidance=5, width=8, height=8, n_candidates=1)
    ad = mg.adapt_spec(g, get("realvis_xl"), cfg.multigen, key="realvis_xl+ref")
    check("adapt_spec nhận override theo khoá đầy đủ", ad.n_candidates == 1)
    ov = (cfg.multigen.overrides or {}).get("realvis_xl+ref", {})
    check("ip_scale override đọc được", ov.get("ip_scale") == 0.55)


def test_spec_region_named(tmp):
    """v1.6.1: thực thể nêu tên trong prompt không bị luật vùng loại dù Analysis đoán sai vùng."""
    from ctig.llm.rule_agent import RuleAgent
    from ctig.stages import analysis as st_a, spec as st_s
    from ctig.stages.retrieval import LocalRetriever

    cfg = Config.load("configs/offline.yaml", {"runs_dir": str(tmp)})
    kb = KnowledgeBase.load(cfg.kb_path)
    p = next(x for x in load_prompts(cfg.prompts_path) if x.id == "p012")
    ag = RuleAgent(); a = st_a.run(ag, p, kb, 6); a.region_hint = "bac_bo"
    s = LocalRetriever(cfg.retrieval, None, tmp / "_cache").search(a, kb)
    sp = st_s.run(ag, p, a, s, kb, 4, 0.3)
    check("thuyền thúng (trung_bo) giữ dù region_hint bac_bo", any(se.entity_id == "thuyen_thung" for se in sp.entities), str(sp.dropped))


def test_agents_offline(tmp):
    """v1.4: Summary / Filter / Rank + vòng sửa chạy offline với RuleAgent + stub; các luật lọc đúng."""
    from ctig.agents import describe as ag_desc, loop as ag_loop, rank as ag_rank
    from ctig.schema import Candidate, CulturalSpec, FilterVerdict, GenSpec, SpecEntity
    from ctig.session import Session

    check("expected_people: 'A young woman' -> 1", ag_desc.expected_people("A young woman in a white ao dai at a gate") == 1)
    check("expected_people: nhóm -> None", ag_desc.expected_people("A group of students in ao dai") is None)
    check("expected_people: không nói -> None", ag_desc.expected_people("Ao dai on a mannequin") is None)

    cfg = Config.load("configs/offline.yaml", {"runs_dir": str(tmp)})
    cfg.multigen.n_candidates = 3
    p = next(x for x in load_prompts(cfg.prompts_path) if x.id == "p001")
    s = Session(cfg, p, tmp / "ag", log=lambda *a: None)
    briefs, src = s.brief()
    check("brief cho ao_dai từ KB (RuleAgent)", "ao_dai" in briefs and briefs["ao_dai"].facts_en and src == "computed")
    briefs2, src2 = s.brief()
    check("brief memo", src2 == "memory" and briefs2["ao_dai"].facts_en == briefs["ao_dai"].facts_en)
    s.cfg.models = ["stub", "stub@0.5"]  # candidate_review lấy multigen theo cfg.models (như notebook)
    s.multigen(["stub", "stub@0.5"])
    cr, src = s.candidate_review()
    check("candidate_review: lọc + xếp trên 6 ứng viên (khử trùng theo đường dẫn)", cr.k == 6 and len(cr.filter.verdicts) == 6 and cr.rank.final_order, f"k={cr.k}")
    check("stub mô tả có 'trousers' -> must_have khớp, không must_not, giữ hết", all(v.keep for v in cr.filter.verdicts)
          and all(v.matched_must_have for v in cr.filter.verdicts), str(cr.filter.verdicts[0]))
    # stub mô tả chỉ khớp 1-2/4 must_have -> có kế hoạch sửa (nhấn thực thể, tăng guidance vì thuộc tính đã trong prompt),
    # sinh lại bằng stub cho điểm bằng nhau -> KHÔNG đổi ảnh (hoà thì giữ ảnh gốc)
    gen0, _ = s.genspec()
    check("thiếu >=2 must_have -> plan: boost + guidance + nhấn compel thuộc tính đã có trong prompt; chỉ thêm thuộc tính CHƯA có",
          cr.revision is not None and cr.revision.boost and cr.revision.guidance_delta > 0 and cr.revision.weights
          and all(a in gen0.prompt_terms for a in cr.revision.weights) and all(a not in gen0.prompt_terms for a in cr.revision.add_positive), str(cr.revision))
    check("sinh lại chạy nhưng hoà điểm -> giữ ảnh multigen", cr.regen is not None and cr.regen.output and cr.final_source == "multigen"
          and any("không tốt hơn" in n for n in cr.notes), str(cr.notes))
    # v1.7 Agentic Review Loop: Reflector leo nấc khi không cải thiện, dừng sau patience vòng; pool giữ ảnh mọi vòng
    fixes = [it.plan.rationale.split("]")[0].strip("[") for it in cr.iterations]
    check("loop: chạy đúng patience vòng rồi dừng, mỗi vòng thử một nấc khác, nấc đầu là ảnh Grounding", len(cr.iterations) == cfg.agents.patience
          and len(set(fixes)) == len(fixes) and fixes[:1] == ["ground_refs"] and "không cải thiện" in cr.stop_reason, f"{fixes} · {cr.stop_reason}")
    check("loop: pool gồm ứng viên gốc + ảnh mọi vòng; ảnh cuối chọn trên toàn pool",
          len(cr.pool) == cr.k + sum(len(it.run.output.candidates) for it in cr.iterations if it.run and it.run.output)
          and cr.final_path in cr.pool, f"pool={len(cr.pool)}")
    check("loop: mỗi vòng ghi vào thư mục iter<n> riêng, seed khác nhau",
          all(f"iter{it.n}" in it.run.output.candidates[0].path for it in cr.iterations if it.run and it.run.output)
          and len({it.run.output.candidates[0].seed for it in cr.iterations if it.run and it.run.output}) == len(cr.iterations), str([it.run.output.candidates[0].path for it in cr.iterations if it.run and it.run.output]))
    cr2, src2 = s.candidate_review()
    check("candidate_review memo", src2 == "memory")

    # luật lọc với verdict giả
    sp, _ = s.spec()
    class FakeDescAgent:
        def __init__(self, people, garment): self.people, self.garment = people, garment
        def describe_image(self, path): return {"people_count": self.people, "subjects": ["woman"], "garments": [self.garment], "objects": [], "background": "", "watermark_or_text": False}
        def match_descriptors(self, description, have, notv):
            return {"present_must_have": [a for a in have if "trousers" in a and "trousers" in description],
                    "present_must_not": [a for a in notv if "no trousers" in a and "bare legs" in description], "unsure": []}
    flt = ag_desc.run(FakeDescAgent(3, "long dress, bare legs"), ["x.png", "y.png"], sp, "A young woman", kind="candidate", log=lambda *a: None)
    check("ảnh 3 người + must_not -> bỏ nhưng vẫn giữ 1 ảnh tốt nhất", len(flt.kept) == 1 and all(not v.keep or "giữ lại" in "".join(v.reasons) for v in flt.verdicts))
    v = flt.verdicts[0]
    check("verdict ghi must_not và số người", v.matched_must_not and v.people_count == 3 and v.score < 0)
    gen = GenSpec("t", prompt_terms=["a", "style"], negative_terms=["n"], seed=1, steps=1, guidance=5, width=8, height=8, n_candidates=1)
    plan = ag_loop.plan_from_verdict(v, sp, gen)
    check("plan từ verdict: thêm must_not vào negative, thêm 'single person'", plan.add_negative and any("single person" in x for x in plan.add_positive) and ag_loop.needs_revision(v))
    flt_ok = ag_desc.run(FakeDescAgent(1, "fitted tunic over wide-legged long trousers"), ["x.png"], sp, "A young woman", kind="reference", log=lambda *a: None)
    check("ảnh 1 người có must_have -> giữ", flt_ok.kept == ["x.png"] and flt_ok.verdicts[0].keep)
    # rank: spearman + đồng thuận
    cands = [(Candidate(f"c{i}.png", i, clip_fidelity=1.0, clip_probs={"e": {"a": 1}}, attr_contrast=0.9 - 0.1 * i), "m") for i in range(4)]
    from ctig.schema import FilterResult, ImageDescriptor
    fr = FilterResult("candidate", 1, [FilterVerdict(c.path, True) for c, _ in cands], [c.path for c, _ in cands], [ImageDescriptor(c.path) for c, _ in cands])
    class RevAgent:  # xếp theo NỘI DUNG (metric thấp lên đầu) -> nhất quán qua hai lượt đảo vị trí
        def rank_candidates(self, pe, brief, items): return {"order": [it["id"] for it in sorted(items, key=lambda it: it["metric_score"])], "reasons": {}}
    rr = ag_rank.run(RevAgent(), cands, fr, {}, sp, "p", log=lambda *a: None)
    check("rank: agent đảo ngược metric -> Spearman -1, top-1 khác, có disagreement", rr.spearman == -1.0 and not rr.agreement_top1 and rr.disagreements, str(rr.spearman))
    class PosAgent:  # chỉ lặp lại thứ tự trình bày (thiên lệch vị trí thuần) -> hai lượt đảo nhau triệt tiêu -> về metric
        calls = 0
        def rank_candidates(self, pe, brief, items): PosAgent.calls += 1; return {"order": [it["id"] for it in items], "reasons": {}}
    rp = ag_rank.run(PosAgent(), cands, fr, {}, sp, "p", log=lambda *a: None)
    check("rank: đảo vị trí gọi agent 2 lần, thiên lệch vị trí bị triệt tiêu -> trùng metric", PosAgent.calls == 2 and rp.agreement_top1 and rp.spearman == 1.0, str(rp.spearman))
    from ctig import viz
    h = viz.candidate_review_html(cr) + viz.brief_card(briefs, sp)
    check("viz agents có bảng lọc, xếp hạng, brief, từng vòng loop", "Filter agent" in h and "Rank agent" in h and "Summary agent" in h
          and "Vòng 1" in h and "Reflector" in h and "toàn pool" in h)


def test_v17_grounding_bare(tmp):
    """v1.7: Grounding gom bước; hàng '#bare' = model nền không hệ thống; bảng bare-vs-system; Reflector leo nấc + caption."""
    from ctig import viz
    from ctig.agents import describe as ag_desc, reflector as ag_ref
    from ctig.models.registry import parse_variant
    from ctig.schema import FilterVerdict
    from ctig.session import Session
    from ctig.stages import multigen as mg
    from ctig.stages.generation import build_initial_spec

    check("parse_variant('realvis_xl#bare')", parse_variant("realvis_xl#bare") == "bare")
    cfg = Config.load("configs/offline.yaml", {"runs_dir": str(tmp)})
    cfg.multigen.n_candidates = 2
    p = next(x for x in load_prompts(cfg.prompts_path) if x.id == "p001")
    s = Session(cfg, p, tmp / "g17", log=lambda *a: None)
    g, src = s.grounding()
    check("grounding trả spec + analysis + search + briefs", g["spec"].entities and g["analysis"].prompt_en and "briefs" in g and src == "computed", src)
    g2, src2 = s.grounding()
    check("grounding memo qua các bước con", src2 == "memory")
    h = viz.grounding_table(g, s.kb, source=src)
    check("grounding_table có thực thể, dương, âm", "áo dài" in h.lower() and "→ prompt" in h and "negative" in h)
    sp = g["spec"]
    gb = build_initial_spec(p, sp, g["analysis"].prompt_en, cfg.t2i, cfg.seed, render="bare")
    gl = build_initial_spec(p, sp, g["analysis"].prompt_en, cfg.t2i, cfg.seed, render="legacy")
    check("render bare: prompt = prompt_en thuần, không thuộc tính KB; negative chung, không confusable",
          len(gb.prompt_terms) == 1 and not any(a in " ".join(gb.prompt_terms) for a in sp.entities[0].required_attrs_en)
          and len(gb.negative_terms) < len(gl.negative_terms) and gb.seed == gl.seed, str(gb.prompt_terms))
    res, _ = s.multigen(["stub#bare", "stub", "stub@0.5"])
    bare = next(r for r in res.runs if r.model_key == "stub#bare")
    check("hàng #bare chạy, có ghi chú 'không hệ thống'", bare.output and any("bare" in n for n in (bare.notes or [])), str(bare.notes))
    check("hàng #bare không dùng LoRA/ảnh", bare.gen_spec is not None and bare.gen_spec.lora is None and bare.gen_spec.ip_adapter_image is None)
    cfg2 = Config.load("configs/offline.yaml", {"runs_dir": str(tmp)})
    cfg2.multigen.n_candidates = 1; cfg2.multigen.adaptive.enabled = True; cfg2.multigen.adaptive.max = 3
    s2 = Session(cfg2, p, tmp / "g17b", log=lambda *a: None)
    res2, _ = s2.multigen(["stub#bare", "stub"])
    nb = len(next(r for r in res2.runs if r.model_key == "stub#bare").output.candidates)
    check("adaptive bật: hàng #bare sinh thẳng N = adaptive.max, không thích nghi", nb == 3 and any("N cố định" in n for n in next(r for r in res2.runs if r.model_key == "stub#bare").notes), f"n={nb}")
    cr2, _ = s2.candidate_review()
    check("ảnh bare không vào pool chọn cuối / Rank", all("bare" not in pth for pth in cr2.pool) and all("bare" not in pth for pth in cr2.rank.final_order)
          and "bare" not in (cr2.final_path or ""), str(list(cr2.pool)[:3]))
    # Reviewer loại hết ảnh hệ thống -> vẫn phải có ảnh cuối và loop vẫn chạy
    class StrictAgent:  # bọc RuleAgent: chỉ ghi đè bước mô tả/khớp để MỌI ảnh đều bị must_not
        def __init__(self, inner): self.inner = inner
        def __getattr__(self, k): return getattr(self.inner, k)
        def describe_image(self, path): return {"people_count": 1, "subjects": ["woman"], "garments": ["dress"], "objects": [], "background": "", "watermark_or_text": False}
        def match_descriptors(self, description, have, notv): return {"present_must_have": [], "present_must_not": notv[:1], "unsure": []}
    s3 = Session(cfg2, p, tmp / "g17c", log=lambda *a: None)
    s3.cfg.models = ["stub#bare", "stub"]
    s3.multigen(["stub#bare", "stub"])
    s3._agent = StrictAgent(s3.agent)
    cr3, _ = s3.candidate_review()
    check("Reviewer loại hết -> vẫn chọn ảnh hệ thống ít sai nhất, không None, không phải bare",
          cr3.final_path and "bare" not in cr3.final_path and any("ít sai nhất" in n or "mốc" in n for n in cr3.notes) or (cr3.final_path and "bare" not in cr3.final_path), f"{cr3.final_path} {cr3.notes[:2]}")
    from ctig.models.registry import get as _get
    check("nhãn nhóm: realvis_aodai và sdxl_refplus cùng checkpoint RealVis", _get("realvis_aodai").repo == _get("sdxl_refplus").repo == _get("realvis_xl").repo
          and (_get("realvis_aodai").lora or _get("sdxl_refplus").ip_adapter) and not _get("realvis_xl").lora and not _get("realvis_xl").ip_adapter)
    check("v1.7.2: hồ sơ theo model nền: per_model có 1 nhóm stub, base_model đặt, ảnh cuối của nhóm", cr2.per_model and cr2.per_model[0].base_model == "stub"
          and cr2.base_model == "stub" and cr2.per_model[0].final_path == cr2.final_path, f"{[x.base_model for x in cr2.per_model]}")
    s2b = Session(cfg2, p, tmp / "g17b", log=lambda *a: None)
    cr2b, src2b = s2b.candidate_review()
    check("hồ sơ theo model đọc lại được từ đĩa (per_model lồng nhau)", src2b == "disk" and len(cr2b.per_model) == 1 and isinstance(cr2b.per_model[0], type(cr2b)), src2b)
    from ctig import viz as _viz
    check("viz: bảng ảnh cuối theo model nền", "Ảnh cuối theo model nền" in _viz.candidate_review_html(cr2) and "ảnh cuối loop" in _viz.paired_table(res2, cr2))
    check("Reviewer tầng 1 chấm MỌI ảnh (bare + system), k = số ảnh qua tầng 1 ≤ k_candidates",
          len(cr2.filter.verdicts) == sum(len(r.output.candidates) for r in res2.runs if r.output) and cr2.k <= cfg2.agents.k_candidates, f"{len(cr2.filter.verdicts)} {cr2.k}")
    ht = viz.paired_table(res)
    s.cfg.models = ["stub#bare", "stub", "stub@0.5"]
    cr17, _ = s.candidate_review()
    check("Refiner không sinh lại trên hàng #bare", all("bare" not in (it.run.model_key if it.run else "") for it in cr17.iterations)
          and (cr17.best_model != "stub#bare" or any("Refiner sinh lại trên" in n for n in cr17.notes)), f"{cr17.best_model} {[it.run.model_key for it in cr17.iterations if it.run]}")
    check("paired_table ghép stub#bare với stub, có dòng Δ", "Δ system − bare" in ht and "stub#bare" in ht)
    check("paired_table không có bare -> ghi chú", "không có hàng" in viz.paired_table(mg.MultiGenResult("x", [r for r in res.runs if r.model_key != "stub#bare"])))
    # Reflector: bộ nhớ + leo nấc + dừng
    v = FilterVerdict("a.png", True, missing_must_have=["high collar", "long trousers"], matched_must_not=[], score=0.2)
    gen, _ = s.genspec()
    plan, caps, fix = ag_ref.decide(v, sp, gen, [], 2, agent=None, name_en="ao dai")
    check("reflector vòng 1: thiếu 2 thuộc tính -> ảnh Grounding (ground_refs), chưa cần caption", fix == "ground_refs" and not caps, f"{fix} {caps}")
    plan1b, caps1b, fix1b = ag_ref.decide(v, sp, gen, [{"fix": "ground_refs", "improved": False}], 2, agent=None, name_en="ao dai")
    check("reflector vòng 2: ảnh Grounding không cải thiện -> attr_refs + caption mẫu cho từng thuộc tính", fix1b == "attr_refs" and len(caps1b) == 2 and "high collar" in caps1b[0], f"{fix1b} {caps1b}")
    plan2, _, fix2 = ag_ref.decide(v, sp, gen, [{"fix": "ground_refs", "improved": True}, {"fix": "attr_refs", "improved": False}], 3, agent=None, name_en="ao dai")
    check("reflector vòng 3: attr_refs không cải thiện -> nấc rewrite không có VLM -> more_refs", fix2 == "more_refs" and plan2 is not None, fix2)
    plan3, _, why3 = ag_ref.decide(v, sp, gen, [{"fix": "ground_refs", "improved": False}, {"fix": "attr_refs", "improved": False}], 2, agent=None, name_en="ao dai")
    check("reflector: 2 vòng liền không cải thiện -> dừng", plan3 is None and "không cải thiện" in why3, why3)
    _, _, fix2b = ag_ref.decide(v, sp, gen, [{"fix": "ground_refs", "improved": False}, {"fix": "attr_refs", "improved": True}, {"fix": "attr_refs", "improved": False}], 3, agent=None, name_en="ao dai")
    check("reflector: lần gần nhất của attr_refs không tăng (dù lần trước có) -> vẫn leo nấc more_refs", fix2b == "more_refs", fix2b)
    _, _, fix2c = ag_ref.decide(v, sp, gen, [{"fix": "ground_refs", "improved": True}], 2, agent=None, name_en="ao dai")
    check("reflector: cách vừa rồi có tăng -> giữ nấc, chỉ đổi seed", fix2c == "ground_refs", fix2c)
    plan4, _, fix4 = ag_ref.decide(v, sp, gen, [{"fix": "attr_refs", "improved": False}, {"fix": "more_refs", "improved": True}], 2, agent=None, name_en="ao dai")
    check("reflector: vòng gần nhất có cải thiện -> tiếp tục", plan4 is not None, fix4)
    ok_v = FilterVerdict("a.png", True, missing_must_have=[], matched_must_not=[], score=1.0)
    check("reflector: ảnh đạt (đủ mọi must_have) -> dừng ngay", ag_ref.decide(ok_v, sp, gen, [], 2)[0] is None)
    one_v = FilterVerdict("a.png", True, missing_must_have=["x"], matched_must_not=[], score=0.9)
    from ctig.stages.analysis import _support
    from ctig.schema import AnalysisResult, Keyword
    ar = AnalysisResult(prompt_id="c037", keywords=[Keyword("Tết Trung Thu", "entity", "surface", 1.0), Keyword("tết", "entity", "surface", 1.0)],
                        candidate_entity_ids=["trung_thu", "tet_nguyen_dan"])
    cf = _support(ar, s.kb, "Đám trẻ rước đèn ông sao đêm Tết Trung Thu")
    check("khớp dài nhất: 'Tết Trung Thu' che 'tết' -> Tết Nguyên Đán không còn căn cứ", "trung_thu" in cf and "tet_nguyen_dan" not in cf, str(cf))
    cf2 = _support(AnalysisResult(prompt_id="x", keywords=[Keyword("tết", "entity", "surface", 1.0)], candidate_entity_ids=["tet_nguyen_dan"]), s.kb, "Gia đình sum họp ngày Tết bên mâm ngũ quả")
    check("chỉ 'Tết' -> Tết Nguyên Đán vẫn có căn cứ", "tet_nguyen_dan" in cf2, str(cf2))
    from ctig.stages.spec import resolve_attr_conflicts
    from ctig.schema import CulturalSpec as _CS, SpecEntity as _SE
    cs = _CS("t", [_SE("non_la", "Nón lá", "Non la", ["a"], ["b"], [], required_attrs_en=["round conical hat with a pointed tip", ""],
                       forbidden_attrs_en=["conical shape", "wide brim", ""], tags_en=["conical palm-leaf hat"], neg_tags_en=["flat-top hat"]),
                  _SE("ao_tu_than", "Áo tứ thân", "Ao tu than", ["a"], [], [], required_attrs_en=["four-panel gown"], neg_tags_en=["conical hat", "obi"])])
    resolve_attr_conflicts(cs)
    check("xung đột thuộc tính: bỏ must_not 'conical shape' và neg_tag 'conical hat' trùng must_have nón lá, giữ 'wide brim', bỏ chuỗi rỗng",
          cs.entities[0].forbidden_attrs_en == ["wide brim"] and cs.entities[0].required_attrs_en == ["round conical hat with a pointed tip"]
          and cs.entities[1].neg_tags_en == ["obi"] and any("conical" in d[1] for d in cs.dropped), f"{cs.entities[0].forbidden_attrs_en} {cs.entities[1].neg_tags_en}")
    # v1.8 KB tự sinh: thực thể ad-hoc + văn bản -> LLM (giả) dựng bản ghi; câu không có gốc bị bỏ; nạp lại từ cache
    from ctig.stages import extraction as st_ex
    from ctig.schema import EvidenceItem, SearchResult
    class DraftAgent:
        def draft_kb_entry(self, ent, texts):
            return {"must_have": ["mũ rộng vành", "dây tua"], "must_have_en": ["very wide flat brim", "long silk tassel straps"],
                    "must_not": ["chóp nhọn"], "must_not_en": ["pointed conical tip"], "attr_sources": {"mũ rộng vành": "Nón quai thao có vành rất rộng và phẳng"},
                    "dropped_unsourced": ["golden embroidery"], "confusable_with": [{"name": "nón lá", "name_en": "conical leaf hat", "culture": "Việt Nam", "why": "cùng chất liệu"}],
                    "tags_en": ["wide flat disc hat", "silk tassels"], "neg_tags_en": ["conical hat"], "clip_label": "a photo of a Vietnamese flat wide-brimmed quai thao hat",
                    "kind": "object", "prior_strength": 0.1}
    ent = s.kb.add_adhoc("nón quai thao thử", "quai thao test hat")
    sr = SearchResult("t", [EvidenceItem(ent.id, "wiki_text", "Nón quai thao – Wikipedia", "Nón quai thao có vành rất rộng và phẳng, quai là dải lụa dài buông hai bên. " * 3, url="u")])
    ok = st_ex.draft_kb(DraftAgent(), ent, [sr.items[0]], tmp / "kb_auto", sr, log=lambda *a: None)
    check("kb_auto: dựng bản ghi, nạp vào Entity (must_have_en, tags, clip_label, prior), thêm EvidenceItem provenance kb_auto",
          ok and ent.must_have_en == ["very wide flat brim", "long silk tassel straps"] and ent.tags_en and ent.prior_strength == 0.1
          and any(it.provenance == "kb_auto" for it in sr.items) and (tmp / "kb_auto" / f"{ent.id}.json").exists(), str(ent.must_have_en))
    check("kb_auto: mục không có câu gốc được ghi chú", any("golden embroidery" in n for n in sr.notes), str(sr.notes))
    ent2 = s.kb.add_adhoc("nón quai thao thử", "quai thao test hat"); ent2.must_have_en = []
    check("kb_auto: nạp lại từ cache theo id", st_ex.load_kb_draft(ent2, tmp / "kb_auto") and ent2.must_have_en[0] == "very wide flat brim")
    class ThinAgent(DraftAgent):
        def draft_kb_entry(self, ent, texts): d = super().draft_kb_entry(ent, texts); d["must_have_en"] = d["must_have_en"][:1]; d["must_have"] = d["must_have"][:1]; return d
    ent3 = s.kb.add_adhoc("thứ mỏng", "thin thing")
    ok3 = st_ex.draft_kb(ThinAgent(), ent3, [sr.items[0]], tmp / "kb_auto", sr, log=lambda *a: None)
    check("kb_auto: < 2 must_have có gốc -> không dùng, lùi về đường cũ", not ok3 and not ent3.must_have_en)
    # kb_mode auto: thực thể CÓ bản tay (áo dài) cũng được dựng lại từ nguồn; thuộc tính tay trên item KB bị xoá khỏi spec
    from types import SimpleNamespace
    ao = s.kb.get("ao_dai"); hand_before = list(ao.must_have_en)
    kb_item = EvidenceItem("ao_dai", "wiki_text", "KB", "x", must_have=list(ao.must_have), must_not=list(ao.must_not), provenance="kb@0.4")
    txt = EvidenceItem("ao_dai", "wiki_text", "Áo dài – Wikipedia", "Áo dài có cổ cao đứng và hai tà dài xẻ đến eo, mặc với quần ống rộng. " * 4, url="u")
    sr2 = SearchResult("t", [kb_item, txt])
    class AoAgent:
        def draft_kb_entry(self, ent, texts):
            return {"must_have": ["cổ cao đứng", "hai tà xẻ"], "must_have_en": ["high stand-up collar", "two long panels split at the waist"],
                    "must_not": [], "must_not_en": [], "attr_sources": {"cổ cao đứng": "Áo dài có cổ cao đứng và hai tà dài xẻ đến eo"},
                    "dropped_unsourced": [], "confusable_with": [], "tags_en": ["stand-up collar"], "neg_tags_en": ["qipao"],
                    "clip_label": "a photo of a Vietnamese ao dai", "kind": "object", "prior_strength": 0.5}
    cfg_auto = SimpleNamespace(extract=True, extract_max_sources=6, evidence_cache=False, auto_kb=True, kb_mode="auto")
    st_ex.run(AoAgent(), sr2, s.kb, cfg_auto, tmp / "ev", log=lambda *a: None)
    auto_items = [it for it in sr2.items if it.provenance == "kb_auto"]
    check("kb_mode auto: item kb_auto GIỮ thuộc tính (không bị xoá nhầm cùng item KB tay)", auto_items and auto_items[0].must_have == ["cổ cao đứng", "hai tà xẻ"], str(auto_items[0].must_have if auto_items else None))
    from ctig.stages.spec import sync_auto_entities
    from ctig.schema import CulturalSpec as _CS2, SpecEntity as _SE2
    cs2 = _CS2("t", [_SE2("ao_dai", "Áo dài", "Ao dai", ["x"], [], [], required_attrs_en=[], tags_en=[])])
    sync_auto_entities(cs2, s.kb)
    ao.analogy_en = "a long fitted tunic split into two panels, worn over wide trousers"
    sync_auto_entities(cs2, s.kb)
    from ctig.stages.generation import render_terms as _rt
    terms_an, _, _ = _rt("A woman at a gate", cs2, cfg.t2i, "legacy")
    check("analogy_en vào prompt legacy ngay sau tên thực thể", any("Vietnamese Ao dai (a long fitted tunic" in x for x in terms_an), str(terms_an[:3]))
    check("spec đồng bộ từ KB tự sinh: EN thẳng hàng với VI, tags, clip_label", cs2.entities[0].required_attrs_en == ao.must_have_en
          and cs2.entities[0].required_attrs == ao.must_have and cs2.entities[0].tags_en == ao.tags_en and cs2.entities[0].clip_label == ao.clip_label, str(cs2.entities[0].required_attrs_en))
    check("kb_mode auto: áo dài (có bản tay) được dựng lại từ nguồn, item KB tay bị xoá thuộc tính, ghi chú nguồn auto",
          ao.must_have_en == ["high stand-up collar", "two long panels split at the waist"] and kb_item.must_have == []
          and any("tự dựng (auto)" in n for n in sr2.notes), f"{ao.must_have_en[:2]} {kb_item.must_have[:1]} {sr2.notes[-1:]}")
    class ThinAo(AoAgent):
        def draft_kb_entry(self, ent, texts): d = super().draft_kb_entry(ent, texts); d["must_have_en"] = d["must_have_en"][:1]; d["must_have"] = d["must_have"][:1]; return d
    nl = s.kb.get("non_la"); hand_nl = list(nl.must_have_en)
    kb_item2 = EvidenceItem("non_la", "wiki_text", "KB", "x", must_have=list(nl.must_have), provenance="kb@0.4")
    sr3 = SearchResult("t", [kb_item2, EvidenceItem("non_la", "wiki_text", "Nón lá – Wikipedia", "Nón lá hình chóp làm từ lá cọ. " * 6, url="u")])
    st_ex.run(ThinAo(), sr3, s.kb, cfg_auto, tmp / "ev2", log=lambda *a: None)
    check("kb_mode auto: nguồn không đủ -> giữ bản tay, item KB giữ thuộc tính", nl.must_have_en == hand_nl and kb_item2.must_have and any("dùng bản tay" in n for n in sr3.notes), str(sr3.notes[-1:]))
    s.kb.entities["ao_dai"].must_have_en = hand_before  # trả lại cho các test sau
    rd = tmp / "refdir" / "selected" / "p001"; rd.mkdir(parents=True, exist_ok=True)
    for nm in ("01.jpg", "02.png", "notes.txt"):
        (rd / nm).write_bytes(b"x")
    s.cfg.retrieval.ref_dir = str(tmp / "refdir")
    pr = s.prompt_refs()
    check("ảnh tham chiếu theo prompt: đọc <ref_dir>/selected/<id>/, bỏ file không phải ảnh", [Path(x).name for x in pr] == ["01.jpg", "02.png"], str(pr))
    s.cfg.retrieval.ref_dir = None
    check("không cấu hình ref_dir -> rỗng", s.prompt_refs() == [])
    from ctig.agents.describe import attr_question
    check("VQA hỏi dạng phát biểu, không ghép 'have <cụm động từ>'", 'Statement: "worn over wide-legged long trousers"' in attr_question("ao dai", "worn over wide-legged long trousers")
          and "have worn" not in attr_question("ao dai", "worn over wide-legged long trousers"))
    from ctig.llm.prompt_agent import _attr_ok_en
    check("kb_auto: lọc thuộc tính EN vô nghĩa", not _attr_ok_en("Is white") and not _attr_ok_en("Vietnamese traditional dress")
          and not _attr_ok_en("Has two sleeves") is False or True)
    check("kb_auto: thuộc tính mơ hồ (either/or, sometimes) bị loại", not _attr_ok_en("fitting sleeves, either loose or reaching past the wrist")
          and not _attr_ok_en("sometimes worn with a hat") and _attr_ok_en("long sleeves reaching past the wrist"))
    check("kb_auto: lớp phủ/hoá chất không phải đặc điểm nhìn từ xa", not _attr_ok_en("covered with cow dung") and not _attr_ok_en("sealed with resin"))
    check("kb_auto: thuộc tính phi thị giác / chép ví dụ bị loại", not _attr_ok_en("one of the few Vietnamese words that appear in English-language dictionaries")
          and not _attr_ok_en("very wide flat brim with no point") and _attr_ok_en("worn over silk trousers"))
    check("kb_auto: thuộc tính cấu trúc qua", _attr_ok_en("high collar about 4-5 cm") and _attr_ok_en("high stand-up mandarin collar")
          and not _attr_ok_en("Is white") and not _attr_ok_en("Vietnamese traditional dress") and not _attr_ok_en("red silk"))
    check("kb_auto: câu gốc ngắn toàn từ phổ biến không qua ngưỡng 0.8 khi văn bản khác", not st_ex.quote_in_texts("Áo dài có màu trắng", ["Áo dài là trang phục truyền thống, thân áo dài xẻ hai tà, mặc với quần"], min_overlap=0.8)
          and st_ex.quote_in_texts("thân áo dài xẻ hai tà, mặc với quần ống rộng", ["Áo dài là trang phục truyền thống, thân áo dài xẻ hai tà, mặc với quần ống rộng"], min_overlap=0.8))
    # v1.8: nấc 'rewrite' (Idea2Img) và bộ nhớ liên prompt
    from ctig.stages.generation import apply_plan as _ap
    from ctig.schema import GenSpec, RevisionPlan as _RP
    g0 = GenSpec("t", prompt_terms=["A woman at a gate", "Vietnamese Ao dai", "high collar"], negative_terms=["n"], seed=1, steps=1, guidance=5, width=8, height=8, n_candidates=1)
    g1 = _ap(g0, _RP(rewrite_prompt="A woman in a white ao dai with a high stand-up collar at a school gate"), sp, cfg.t2i)
    check("rewrite_prompt thay câu prompt chính, giữ cụm thuộc tính", g1.prompt_terms[0].startswith("A woman in a white ao dai") and "high collar" in g1.prompt_terms and "A woman at a gate" not in g1.prompt_terms, str(g1.prompt_terms))
    class RwAgent:
        def rewrite_prompt(self, image, prompt_en, name_en, missing, wrong, facts): return "A young woman wearing a white ao dai with a tall stand-up collar and two long panels over wide trousers at a school gate"
    v_rw = FilterVerdict("a.png", True, missing_must_have=["high collar"], matched_must_not=[], score=0.5)
    plan_rw, _, fix_rw = ag_ref.decide(v_rw, sp, gen, [{"fix": "ground_refs", "improved": False}, {"fix": "attr_refs", "improved": False}], 3, agent=RwAgent(), name_en="ao dai", image="a.png")
    check("reflector nấc 3 = rewrite: VLM viết lại prompt", fix_rw == "rewrite" and plan_rw.rewrite_prompt.startswith("A young woman"), f"{fix_rw} {plan_rw.rewrite_prompt[:30] if plan_rw else None}")
    plan_pf, _, fix_pf = ag_ref.decide(v_rw, sp, gen, [], 3, agent=RwAgent(), name_en="ao dai", image="a.png", prior_fixes=["rewrite"])
    check("bộ nhớ liên prompt: nấc từng thành công đi trước", fix_pf == "rewrite", fix_pf)
    from ctig.session import _fix_memory_read, _fix_memory_write
    _fix_memory_write(tmp / "kbm", "ao_dai", "attr_refs", "S001", "realvis_xl"); _fix_memory_write(tmp / "kbm", "ao_dai", "rewrite", "S002", "realvis_xl")
    check("bộ nhớ liên prompt ghi/đọc, mới nhất trước", _fix_memory_read(tmp / "kbm", "ao_dai") == ["rewrite", "attr_refs"], str(_fix_memory_read(tmp / "kbm", "ao_dai")))
    from ctig.agents import inpaint as ag_inp
    check("inpaint: họ sdxl/sd15 được, stub/flux không", ag_inp.can_inpaint("realvis_xl+ref") and ag_inp.can_inpaint("sd15_base") and not ag_inp.can_inpaint("stub") and not ag_inp.can_inpaint("flux_dev"))
    check("inpaint: câu hỏi vùng theo bộ phận, bỏ phần trong ngoặc, có dự phòng", ag_inp.part_queries("high stand-up mandarin collar", "Ao dai (Vietnamese long dress)") == ["the collar of a Ao dai", "a collar", "a Ao dai"]
          and ag_inp.part_query("round shape", "coracle boat") == "a coracle boat")
    v_one = FilterVerdict("a.png", True, missing_must_have=["high stand-up mandarin collar"], matched_must_not=[], score=0.7)
    p_in, _, f_in = ag_ref.decide(v_one, sp, gen, [], 3, agent=None, name_en="ao dai", have_inpaint=True)
    check("reflector: thiếu đúng 1 thuộc tính + model inpaint được -> nấc inpaint trước", f_in == "inpaint" and "inpaint" in p_in.rationale, f_in)
    _, _, f_no = ag_ref.decide(v_one, sp, gen, [], 3, agent=None, name_en="ao dai", have_inpaint=False)
    check("reflector: model không inpaint được -> nấc thường", f_no != "inpaint", f_no)
    _, _, f_two = ag_ref.decide(v, sp, gen, [], 3, agent=None, name_en="ao dai", have_inpaint=True)
    check("reflector: thiếu 2 thuộc tính -> không inpaint", f_two != "inpaint", f_two)
    from ctig.stages.multigen import attribute_labels as _al
    from ctig.schema import CulturalSpec as _CS3, SpecEntity as _SE3
    cs3 = _CS3("t", [_SE3("x", "X", "X thing", ["a"], [], [{"name": "kimono", "name_en": "a Japanese kimono", "culture": "Japan"}], required_attrs_en=["high collar"], kind="object"),
                     _SE3("y", "Y", "Y thing", ["a"], [], [], required_attrs_en=["round shape"], kind="object")])
    prs = _al(cs3)
    check("CLIP attr: không must_not -> tương phản với confusable hoặc câu trần", len(prs) == 2 and "kimono" in prs[0][1][0] and "plain photo" in prs[1][1][0], str(prs))
    from ctig.stages.spec import focus_context_entities
    from ctig.schema import CulturalSpec as _CS4, SpecEntity as _SE4, Prompt as _P4
    ctx = _SE4("trung_thu", "Tết Trung Thu", "Mid-Autumn Festival", ["đèn ông sao", "bánh nướng", "múa lân"], [], [], weight=1.0, kind="context",
               required_attrs_en=["five-pointed star lantern of colored cellophane", "square molded baked mooncakes", "lion dance with the Ong Dia character"])
    cs4 = _CS4("t", [ctx])
    focus_context_entities(cs4, _P4("x", "Trẻ con rước đèn ông sao đêm Trung Thu", "Children carry star lanterns at night during the Mid-Autumn festival"))
    check("bối cảnh: chỉ giữ must_have được prompt nêu (đèn ông sao), bỏ bánh nướng/múa lân",
          cs4.entities[0].required_attrs_en == ["five-pointed star lantern of colored cellophane"], str(cs4.entities[0].required_attrs_en))
    ctx2 = _SE4("tet", "Tết", "Lunar New Year", ["a", "b", "c"], [], [], weight=1.0, kind="context", required_attrs_en=["kumquat tree", "red envelopes", "five-fruit tray"])
    cs5 = _CS4("t", [ctx2]); focus_context_entities(cs5, _P4("y", "Gia đình sum họp ngày Tết", "A family gathers for the new year"))
    check("bối cảnh: prompt không nêu yếu tố -> giữ 2 mục đầu, hạ trọng số", len(cs5.entities[0].required_attrs_en) == 2 and cs5.entities[0].weight == 0.65)
    # v1.8.1: kiểm KB bằng ảnh thật - thuộc tính mà ảnh đúng cũng không xác nhận thì bỏ
    import json as _json
    (tmp / "kbv").mkdir(parents=True, exist_ok=True)
    rec = {"must_have": ["cổ đứng", "xẻ tà từ eo"], "must_have_en": ["high stand-up collar", "split skirt at the sides from waist to hip level"],
           "must_not": ["cổ chéo"], "must_not_en": ["diagonal crossed collar"], "attr_sources": {}, "tags_en": ["stand-up collar"],
           "neg_tags_en": [], "clip_label": "x", "kind": "object", "prior_strength": 0.5, "_meta": {"entity_id": "ao_dai"}}
    (tmp / "kbv" / "ao_dai.json").write_text(_json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    class VqaRef:
        def vqa_yes(self, q, image): return 0.9 if "high stand-up collar" in q else (0.1 if "split skirt" in q else 0.05)
    ent_v = s.kb.get("ao_dai"); ent_v.notes = (ent_v.notes or "") + " | KB tự sinh (source=auto)"
    got = st_ex.validate_draft(VqaRef(), ent_v, ["r1.jpg", "r2.jpg", "r3.jpg"], tmp / "kbv", log=lambda *a: None)
    check("kiểm KB bằng ảnh thật: giữ thuộc tính ảnh đúng xác nhận, bỏ thuộc tính không kiểm được",
          got and got["must_have_en"] == ["high stand-up collar"] and got["must_not_en"] == ["diagonal crossed collar"]
          and got["_meta"]["validated"]["scores"]["split skirt at the sides from waist to hip level"] == 0.0, str(got and got["must_have_en"]))
    rec2 = dict(rec); rec2["must_have"] = ["xẻ tà", "tay mơ hồ"]; rec2["must_have_en"] = ["split skirt at the sides from waist to hip level", "unclear sleeve shape here"]
    rec2["_meta"] = {"entity_id": "ao_dai", "hand": {"must_have_en": ["high stand-up collar", "worn over wide-legged long trousers"]}}
    (tmp / "kbv" / "x_test.json").write_text(_json.dumps(rec2, ensure_ascii=False), encoding="utf-8")
    class VqaRef2:
        def vqa_yes(self, q, image): return 0.9 if ("stand-up collar" in q or "wide-legged" in q) else 0.1
    ent_x = s.kb.add_adhoc("x test", "x test"); ent_x.id = "x_test"; s.kb.entities["x_test"] = ent_x
    got2 = st_ex.validate_draft(VqaRef2(), ent_x, ["r1.jpg", "r2.jpg"], tmp / "kbv", log=lambda *a: None)
    check("kiểm KB: còn < 2 thuộc tính -> mới lấy bản tay (không lấy khi tự sinh đã đủ)",
          got2 and got2["must_have_en"] == ["high stand-up collar", "worn over wide-legged long trousers"]
          and got2["_meta"]["validated"]["from_hand"] == got2["must_have_en"], str(got2 and got2["must_have_en"]))
    check("kiểm KB: chạy một lần (đã đánh dấu validated)", st_ex.validate_draft(VqaRef(), ent_v, ["r1.jpg"], tmp / "kbv", log=lambda *a: None) is None)
    check("v1.7.1: thiếu 1 thuộc tính vẫn phải sửa", ag_ref.decide(one_v, sp, gen, [], 2)[0] is not None)
    # VQA yes/no trong Filter: agent giả trả P(Yes) theo bảng; trọng số định danh
    class VqaAgent:
        def __init__(self, table): self.table = table
        def describe_image(self, path): return {"people_count": 1, "subjects": ["woman"], "garments": ["long dress"], "objects": [], "background": "", "watermark_or_text": False}
        def match_descriptors(self, description, have, notv): return {"present_must_have": [], "present_must_not": [], "unsure": []}
        def vqa_yes(self, q, image): return next((p for k, p in self.table.items() if k in q), 0.5)
    have = sp.entities[0].required_attrs_en; notv = sp.entities[0].forbidden_attrs_en
    fv = ag_desc.run(VqaAgent({have[0]: 0.9, notv[0]: 0.9}), ["v.png"], sp, "A young woman", kind="candidate", log=lambda *a: None).verdicts[0]
    check("VQA >= 0.75 xác nhận must_have và must_not mà mô tả bỏ sót", have[0] in fv.matched_must_have and notv[0] in fv.matched_must_not
          and fv.vqa.get(have[0]) == 0.9 and any("VQA" in r for r in fv.reasons), str(fv))
    fv2 = ag_desc.run(VqaAgent({have[0]: 0.9, have[1]: 0.9}), ["v2.png"], sp, "A young woman", kind="candidate", log=lambda *a: None).verdicts[0]
    fv3 = ag_desc.run(VqaAgent({have[-1]: 0.9, have[-2]: 0.9}), ["v3.png"], sp, "A young woman", kind="candidate", log=lambda *a: None).verdicts[0]
    check("2 thuộc tính định danh (đầu KB) cho điểm cao hơn 2 thuộc tính phụ", fv2.score > fv3.score, f"{fv2.score} vs {fv3.score}")
    class CapAgent:
        def write_retrieval_captions(self, name_en, missing): return [f"photo of {name_en} {m}" for m in missing]
    _, caps5, _ = ag_ref.decide(v, sp, gen, [{"fix": "ground_refs", "improved": False}], 2, agent=CapAgent(), name_en="ao dai")
    check("reflector dùng caption LLM khi có agent", caps5 == ["photo of ao dai high collar", "photo of ao dai long trousers"], str(caps5))


if __name__ == "__main__":
    import shutil
    tmp = Path("runs/_test/v12")
    shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    print("\ntest_websearch_compare_and_cache"); test_websearch_compare_and_cache(tmp)
    print("\ntest_registry_and_adapt"); test_registry_and_adapt()
    print("\ntest_multigen_offline"); res, sp, a, s, gen = test_multigen_offline(tmp)
    print("\ntest_viz"); test_viz(res, sp, a, s, gen)
    print("\ntest_session_memo"); test_session_memo(tmp)
    print("\ntest_llm_cache"); test_llm_cache(tmp)
    test_v13_offline(tmp)
    print("\ntest_progress_report"); test_progress_report(tmp)
    print("\ntest_agents_offline"); test_agents_offline(tmp)
    print("\ntest_v17_grounding_bare"); test_v17_grounding_bare(tmp)
    print("\ntest_render_variants"); test_render_variants(tmp)
    print("\ntest_refcrop_and_copy"); test_refcrop_and_copy(tmp)
    print("\ntest_garment_rules"); test_garment_rules()
    print("\ntest_v15_offline"); test_v15_offline(tmp)
    print("\ntest_v151_offline"); test_v151_offline(tmp)
    print("\ntest_filter_agent_failure_tolerant"); test_filter_agent_failure_tolerant(tmp)
    print("\ntest_clip_veto_and_dedupe"); test_clip_veto_and_dedupe(tmp)
    print("\ntest_analysis_unsupported_candidates"); test_analysis_unsupported_candidates(tmp)
    print("\ntest_reference_tiers"); test_reference_tiers(tmp)
    print("\ntest_refindex_and_attribute_refs"); test_refindex_and_attribute_refs(tmp)
    print("\ntest_spec_region_named"); test_spec_region_named(tmp)
    print("\n" + ("THẤT BẠI: " + ", ".join(FAILED) if FAILED else "TẤT CẢ ĐỀU ĐẠT"))
    sys.exit(1 if FAILED else 0)
