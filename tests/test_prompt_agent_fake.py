"""
Kiểm plumbing của PromptAgent + VLMClipPerceiver với một backend GIẢ trả JSON soạn sẵn.
Không cần GPU. Không kiểm chất lượng model, chỉ kiểm: schema, parse, lọc id bịa, bỏ thuộc tính
không có nguồn, dịch thuộc tính, ánh xạ finding, chạy được từ đầu tới cuối với ảnh giả.
Chạy: python tests/test_prompt_agent_fake.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ctig.config import Config
from ctig.kb import KnowledgeBase
from ctig.llm.base import JSONChatMixin
from ctig.llm.prompt_agent import PromptAgent
from ctig.pipeline import Pipeline, load_prompts
from ctig.schema import Candidate, GenOutput
from ctig.stages.perception import VLMClipPerceiver
from ctig.stages.evaluation import AgentJudge

FAILED = []


def check(name, cond, detail=""):
    print(("  ok    " if cond else "  FAIL  ") + name + ("" if cond else f"  {detail}"))
    if not cond:
        FAILED.append(name)


class FakeBackend(JSONChatMixin):
    """Nhận ra nhiệm vụ qua system prompt và trả JSON tương ứng, có cố ý chèn lỗi để kiểm bộ lọc."""

    name = "fake"
    json_retries = 1

    def __init__(self):
        self.calls = []

    def chat(self, system, user, images=None):
        self.calls.append((system[:40], bool(images)))
        if "agent phân tích prompt" in system:
            return json.dumps({
                "keywords": [{"term": "Áo dài", "kind": "entity", "source": "surface", "confidence": 0.9, "rationale": "nêu tên"}],
                "candidate_entity_ids": ["ao_dai", "id_bia_dat"],
                "region_hint": "sao_hoa", "prompt_en": "A woman in a white ao dai at a school gate", "notes": "",
                "new_entities": [{"name_vi": "Gốm Bát Tràng", "name_en": "Bat Trang ceramics", "category": "sinh_hoat",
                                  "region": "bac_bo", "rationale": "test"}],
            }, ensure_ascii=False)
        if "rút ra bằng chứng" in system:
            # attr thứ hai có quote KHÔNG nằm trong văn bản -> phải bị loại
            return json.dumps({
                "must_have": [{"attr": "cổ đứng cao", "quote": "[0] áo dài có cổ đứng cao ôm sát cổ"},
                              {"attr": "thuộc tính bịa", "quote": "câu này không có trong văn bản nào cả"}],
                "must_not": [{"attr": "đai obi", "quote": "[0] khác kimono ở chỗ không có đai obi"}],
                "confusable_with": [{"name": "kimono", "name_en": "a Japanese kimono with obi", "culture": "Nhật", "why": "gần giống"}],
            }, ensure_ascii=False)
        if "Dịch các đặc điểm thị giác (bắt buộc" in system:
            d = json.loads(user)
            # cố ý nhiễm 'wide obi' vào cụm đầu để kiểm bộ lọc
            return json.dumps({"entities": [{"entity_id": k, "attrs_en": ["long tunic with wide obi at back"] +
                                             [f"EN:{a}" for a in v["attrs"][1:]]} for k, v in d.items()]})
        if "Dịch các đặc điểm thị giác (KHÔNG" in system:
            d = json.loads(user)
            return json.dumps({"entities": [{"entity_id": k, "attrs_en": [f"NEG:{a}" for a in v["attrs"]]} for k, v in d.items()]})
        if "Mô tả ảnh cho hệ thống kiểm tra" in system:
            return json.dumps({"caption": "a woman in a kimono", "elements": [
                {"label": "kimono", "category": "trang_phuc", "attrs": ["đai obi bản rộng"], "confidence": 0.8}]}, ensure_ascii=False)
        if "trả lời các câu hỏi ĐÓNG" in system:
            n_req = user.count("\n  R"); n_forb = user.count("\n  F")
            return json.dumps({"identity": "confusable:kimono", "attrs": ["no"] * n_req, "forbidden": ["yes"] + ["no"] * (n_forb - 1), "note": "fake"})
        if "trọng tài đánh giá" in system:
            return json.dumps({"identity": 0.0, "completeness": 0.0, "purity": 0.0, "score": 0.0, "reasoning": "fake judge"})
        return "{}"


def test_analyze_filters():
    kb = KnowledgeBase.load(Config().kb_path)
    agent = PromptAgent(FakeBackend())
    p = load_prompts(Config().prompts_path)[0]
    from ctig.stages import analysis
    a = analysis.run(agent, p, kb)
    check("id bịa bị lọc", "id_bia_dat" not in a.candidate_entity_ids, str(a.candidate_entity_ids))
    check("region không hợp lệ -> None", a.region_hint is None)
    check("thực thể mới được đăng ký ad-hoc", any(i.startswith("x_gom") for i in a.candidate_entity_ids), str(a.candidate_entity_ids))
    check("KB có thực thể ad-hoc", kb.get("x_gom_bat_trang") is not None)


def test_extract_drops_unsourced():
    kb = KnowledgeBase.load(Config().kb_path)
    agent = PromptAgent(FakeBackend())
    text = "Áo dài có cổ đứng cao ôm sát cổ và xẻ tà hai bên. Nó khác kimono ở chỗ không có đai obi."
    d = agent.extract_evidence(kb.get("ao_dai"), [{"title": "t", "url": "u", "text": text}])
    check("thuộc tính có câu gốc được giữ", d["must_have"] == ["cổ đứng cao"], str(d["must_have"]))
    check("thuộc tính có quote bịa bị bỏ", "thuộc tính bịa" in d["dropped_unsourced"], str(d["dropped_unsourced"]))
    check("confusable có name_en", d["confusable_with"][0].get("name_en", "").startswith("a Japanese"))


def test_end_to_end_with_fake_vlm():
    """Pipeline đầy đủ với agent giả + tri giác giả (không CLIP) + bộ sinh stub."""
    cfg = Config.load("configs/offline.yaml", {"runs_dir": "runs/_test", "run_name": "fake-vlm",
                                               "cache": {"enabled": False},
                                               "retrieval": {"backend": "local", "extract": True}})
    pipe = Pipeline(cfg, log=lambda *a, **k: None)
    fake = FakeBackend()
    pipe.agent = PromptAgent(fake)
    pipe.perceiver = VLMClipPerceiver(fake, clip=None)
    pipe.judge = AgentJudge(pipe.agent)
    p = load_prompts(cfg.prompts_path)[0]
    res = pipe.run_one(p)
    se = res.spec.entity("ao_dai")
    check("spec có Áo dài", se is not None)
    check("cụm dịch nhiễm 'obi' bị loại", se is not None and not any("obi" in a for a in se.required_attrs_en), str(se.required_attrs_en if se else None))
    # v1.2: KB có bản EN viết tay -> không cần dịch; mọi cụm EN phải không rỗng và không nhiễm confusable
    check("EN lấy từ KB viết tay, đủ và không rỗng", se is not None and len(se.required_attrs) == len(se.required_attrs_en)
          and all(se.required_attrs_en), str(se.required_attrs_en if se else None))
    check("forbidden EN từ KB", se is not None and len(se.forbidden_attrs) == len(se.forbidden_attrs_en) and all(se.forbidden_attrs_en))
    check("VLM dịch KHÔNG được gọi khi KB đã có EN", not any("Dịch các đặc điểm" in sys_ for sys_, _ in fake.calls))
    it0 = res.outcome.iterations[0]
    check("checklist được hỏi (có gửi ảnh)", any(img for sys_, img in fake.calls if "câu hỏi ĐÓNG" in sys_))
    check("checklist identity=confusable -> finding critical", any(f.severity == "critical" for f in it0.critiques[0].findings))
    check("verdict revise", it0.adjudication.verdict == "revise")
    check("negative có 'kimono' (từ vòng 0 hoặc bản sửa)",
          any("kimono" in n for n in it0.gen_spec.negative_terms + it0.plan.add_negative), str(it0.gen_spec.negative_terms))
    check("negative KHÔNG chứa 'áo dài' hay 'vietnamese'", not any(("áo dài" in n or "vietnamese" in n.lower()) for n in it0.gen_spec.negative_terms), str(it0.gen_spec.negative_terms))
    check("thuộc tính tiếng Việt gốc không lọt vào prompt", se is not None and not any(a in it0.gen_spec.prompt_terms for a in se.required_attrs), str(it0.gen_spec.prompt_terms))
    it1 = res.outcome.iterations[1] if len(res.outcome.iterations) > 1 else None
    check("nhấn tối đa một lần", it1 is None or it1.gen_spec.prompt_terms.count("Vietnamese Ao dai") <= 1, str(it1.gen_spec.prompt_terms[:4] if it1 else None))
    check("ra ảnh cuối", Path(res.record.final_image_path).exists())


if __name__ == "__main__":
    for fn in (test_analyze_filters, test_extract_drops_unsourced, test_end_to_end_with_fake_vlm):
        print(f"\n{fn.__name__}")
        fn()
    print("\n" + ("THẤT BẠI: " + ", ".join(FAILED) if FAILED else "TẤT CẢ ĐỀU ĐẠT"))
    sys.exit(1 if FAILED else 0)
