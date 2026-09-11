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
    check("playground dùng VAE fp16-fix (tránh upcast fp32 trên T4)", get("playground25").vae == "madebyollin/sdxl-vae-fp16-fix")
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
    print("\n" + ("THẤT BẠI: " + ", ".join(FAILED) if FAILED else "TẤT CẢ ĐỀU ĐẠT"))
    sys.exit(1 if FAILED else 0)
