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
    check("tags: thực thể lên đầu, thẻ ngắn, có trọng số thẻ đầu", g_tags.prompt_terms[0].startswith("Vietnamese") and "fitted long tunic" in g_tags.prompt_terms
          and g_tags.term_weights.get("fitted long tunic") == cfg.t2i.emphasis_weight, str(g_tags.prompt_terms[:4]))
    check("tags: negative dùng neg_tags, không có 'trousers'", "obi sash" in g_tags.negative_terms and not any("trousers" in n for n in g_tags.negative_terms), str(g_tags.negative_terms))
    check("legacy: cảnh trước, câu dài, negative must_not_en", g_leg.prompt_terms[0] == a.prompt_en and any("no trousers" in n for n in g_leg.negative_terms))
    check("sentence: một đoạn văn", len(g_sen.prompt_terms) == 1 and g_sen.prompt_terms[0].startswith(a.prompt_en.rstrip(".")) and "has" in g_sen.prompt_terms[0])
    check("mặc định config = tags", build_initial_spec(p, sp, a.prompt_en, cfg.t2i, 1).render == "tags")
    check("parse_variant", parse_variant("realvis_xl#legacy") == "legacy" and parse_variant("realvis_xl@0.6") is None and parse_key("sdxl_aodai@0.6#tags") == ("sdxl_aodai", 0.6))
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
    check("chép (ref_sim 0.97) bị trừ điểm tổng, 0.70 thì không", mg.combined_score(c_copy) < mg.combined_score(c_ok) - 0.2 and abs(mg.combined_score(c_ok) - 0.9) < 1e-6)
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
    s.multigen(["stub", "stub@0.5"])
    cr, src = s.candidate_review()
    check("candidate_review: lọc + xếp trên 6 ứng viên", cr.k == 6 and len(cr.filter.verdicts) == 6 and cr.rank.final_order)
    check("stub mô tả có 'trousers' -> must_have khớp, không must_not, giữ hết", all(v.keep for v in cr.filter.verdicts)
          and all(v.matched_must_have for v in cr.filter.verdicts), str(cr.filter.verdicts[0]))
    # stub mô tả chỉ khớp 1-2/4 must_have -> có kế hoạch sửa (nhấn thực thể, tăng guidance vì thuộc tính đã trong prompt),
    # sinh lại bằng stub cho điểm bằng nhau -> KHÔNG đổi ảnh (hoà thì giữ ảnh gốc)
    check("thiếu >=2 must_have -> plan: boost + guidance + nhấn compel thẻ đã có trong prompt, không thêm trùng", cr.revision is not None
          and cr.revision.boost and cr.revision.guidance_delta > 0 and cr.revision.weights and not cr.revision.add_positive, str(cr.revision))
    check("sinh lại chạy nhưng hoà điểm -> giữ ảnh multigen", cr.regen is not None and cr.regen.output and cr.final_source == "multigen"
          and any("không tốt hơn" in n for n in cr.notes), str(cr.notes))
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
    class RevAgent:
        def rank_candidates(self, pe, brief, items): return {"order": [it["id"] for it in reversed(items)], "reasons": {}}
    rr = ag_rank.run(RevAgent(), cands, fr, {}, sp, "p", log=lambda *a: None)
    check("rank: agent đảo ngược metric -> Spearman -1, top-1 khác, có disagreement", rr.spearman == -1.0 and not rr.agreement_top1 and rr.disagreements)
    from ctig import viz
    h = viz.candidate_review_html(cr) + viz.brief_card(briefs, sp)
    check("viz agents có bảng lọc, xếp hạng, brief", "Filter agent" in h and "Rank agent" in h and "Summary agent" in h)


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
    print("\ntest_render_variants"); test_render_variants(tmp)
    print("\ntest_refcrop_and_copy"); test_refcrop_and_copy(tmp)
    print("\n" + ("THẤT BẠI: " + ", ".join(FAILED) if FAILED else "TẤT CẢ ĐỀU ĐẠT"))
    sys.exit(1 if FAILED else 0)
