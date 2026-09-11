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
            return json.dumps({
                "must_have": ["cổ đứng cao", "thuộc tính bịa không nguồn"],
                "must_not": ["đai obi"],
                "confusable_with": [{"name": "kimono", "culture": "Nhật", "why": "gần giống"}],
                "attr_sources": {"cổ đứng cao": "[0] áo dài có cổ đứng cao", "đai obi": "[0] khác kimono ở đai obi"},
            }, ensure_ascii=False)
        if "Dịch các thuộc tính" in system:
            d = json.loads(user)
            return json.dumps({"entities": [{"entity_id": k, "required_en": [f"EN:{a}" for a in v["required"]],
                                             "forbidden_en": [f"EN:{a}" for a in v["forbidden"]]} for k, v in d.items()]})
        if "mô tả một ảnh" in system:
            return json.dumps({"caption": "a woman in a kimono", "elements": [
                {"label": "kimono", "category": "trang_phuc", "attrs": ["đai obi bản rộng"], "confidence": 0.8}]}, ensure_ascii=False)
        if "chuyên gia văn hoá vật chất" in system:
            return json.dumps({"findings": [
                {"entity_id": "ao_dai", "severity": "critical", "observed": "kimono có obi", "expected": "Áo dài", "message": "vẽ kimono"},
                {"entity_id": "khong_ton_tai", "severity": "minor", "observed": "x", "expected": "y", "message": "phải bị lọc"},
            ], "score": 0.2, "verdict": "revise", "reasoning": "fake"}, ensure_ascii=False)
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
    d = agent.extract_evidence(kb.get("ao_dai"), [{"title": "t", "url": "u", "text": "x" * 100}])
    check("thuộc tính không có nguồn bị bỏ", d["must_have"] == ["cổ đứng cao"], str(d["must_have"]))
    check("ghi lại thứ đã bỏ", "thuộc tính bịa không nguồn" in d["dropped_unsourced"])


def test_end_to_end_with_fake_vlm():
    """Pipeline đầy đủ với agent giả + tri giác giả (không CLIP) + bộ sinh stub."""
    cfg = Config.load("configs/offline.yaml", {"runs_dir": "runs/_test", "run_name": "fake-vlm",
                                               "cache": {"enabled": False},
                                               "retrieval": {"backend": "local", "extract": True}})
    pipe = Pipeline(cfg, log=lambda *a, **k: None)
    fake = FakeBackend()
    pipe.agent = PromptAgent(fake)
    pipe.perceiver = VLMClipPerceiver(fake, clip=None)
    p = load_prompts(cfg.prompts_path)[0]
    res = pipe.run_one(p)
    se = res.spec.entity("ao_dai")
    check("spec có Áo dài", se is not None)
    check("thuộc tính được dịch EN", se is not None and se.required_attrs_en and se.required_attrs_en[0].startswith("EN:"),
          str(se.required_attrs_en if se else None))
    it0 = res.outcome.iterations[0]
    check("finding với entity_id lạ bị lọc", all(f.entity_id != "khong_ton_tai" for f in it0.critiques[0].findings))
    check("VLM critique critical -> revise", it0.adjudication.verdict == "revise")
    check("bản sửa có negative 'kimono'", any("kimono" in n for n in it0.plan.add_negative), str(it0.plan.add_negative))
    check("critique có nhận ảnh", any(img for sys_, img in fake.calls if "chuyên gia" in sys_))
    check("ra ảnh cuối", Path(res.record.final_image_path).exists())


if __name__ == "__main__":
    for fn in (test_analyze_filters, test_extract_drops_unsourced, test_end_to_end_with_fake_vlm):
        print(f"\n{fn.__name__}")
        fn()
    print("\n" + ("THẤT BẠI: " + ", ".join(FAILED) if FAILED else "TẤT CẢ ĐỀU ĐẠT"))
    sys.exit(1 if FAILED else 0)
