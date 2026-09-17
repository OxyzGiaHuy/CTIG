"""Tiêm câu prompt từ NGOÀI vào pha sinh, mà bảng kiểm (CulturalSpec) phải giữ nguyên.

Đây là điều kiện để ba nhánh A/B/C so sánh được: chỉ CÂU đưa vào bộ sinh khác nhau, còn thước đo bên trong
(must_have/must_not) phải giống hệt. Nếu spec đổi theo câu chữ thì hiệu số B→C không còn đo đúng vòng sửa.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ctig.config import Config
from ctig.schema import Prompt, to_dict
from ctig.session import Session

ROOT = Path(__file__).resolve().parent.parent
CT = ("A young woman wearing a white Vietnamese ao dai, a long-sleeved silk tunic split at the hips into "
      "flowing front and back panels over wide-leg white trousers, stands at an iron school gate.")

with tempfile.TemporaryDirectory() as td:
    cfg = Config.load(str(ROOT / "configs" / "offline.yaml"), {"runs_dir": td, "t2i": {"render": "bare"}})
    p = Prompt("S001", "Một cô gái mặc áo dài trắng đứng trước cổng trường.",
               "A young woman in a white ao dai standing at a school gate.")
    s = Session(cfg, p, run_dir=Path(td) / "run", log=lambda *a: None)

    spec_before, _ = s.spec()
    gen_before, _ = s.genspec()
    assert gen_before.prompt_terms, gen_before.prompt_terms
    print("render=bare, câu gốc ->", gen_before.prompt_terms)

    # khoá memo của các bước grounding KHÔNG được chứa câu prompt, nếu không spec sẽ tính lại theo câu mới
    assert "pe" not in s._base_key() and CT not in str(s._base_key())

    old = s.set_prompt_en(CT)
    assert old == (gen_before.prompt_terms[0] if gen_before.prompt_terms else ""), old

    spec_after, src_spec = s.spec()
    assert src_spec == "memory", f"spec phải lấy từ bộ nhớ, không tính lại (được '{src_spec}')"
    assert to_dict(spec_after) == to_dict(spec_before), "BẢNG KIỂM ĐÃ ĐỔI sau khi thay câu prompt"
    have = [a for e in spec_after.entities for a in e.required_attrs_en]
    print("bảng kiểm giữ nguyên: %d thuộc tính bắt buộc" % len(have))

    gen_after, src_gen = s.genspec()
    assert src_gen == "computed", f"genspec phải tính lại (được '{src_gen}')"
    assert gen_after.prompt_terms == [CT], gen_after.prompt_terms
    print("câu đưa vào bộ sinh đã thay, dài %d từ" % len(gen_after.prompt.split()))

    # gọi lại với cùng chuỗi -> không làm gì, không xoá memo
    s.set_prompt_en(CT)
    assert s.genspec()[1] == "memory"
    # chuỗi rỗng -> bỏ qua
    assert s.set_prompt_en("") == ""
    assert s.genspec()[0].prompt_terms == [CT]
    print("gọi lại cùng chuỗi / chuỗi rỗng: không phá memo")

# --- 2026-09-17: cờ --no-grounding từng nuốt mất câu prompt ngoài -------------------------------
# Session.skip_grounding() dựng AnalysisResult mới ở MỖI lần gọi analysis(), mà set_prompt_en lại sửa
# tại chỗ trên đối tượng đó. Kết quả: nhánh B âm thầm chạy bằng câu GỐC, ảnh trùng BYTE với nhánh A
# ở cả 10 prompt, và lưới so sánh trông như Culture-TRIP không có tác dụng gì.
with tempfile.TemporaryDirectory() as td:
    cfg2 = Config.load(str(ROOT / "configs" / "offline.yaml"), {"runs_dir": td, "t2i": {"render": "bare"}})
    p2 = Prompt("S001", "Một cô gái mặc áo dài trắng đứng trước cổng trường.",
                "A young woman in a white ao dai standing at a school gate.")
    s2 = Session(cfg2, p2, run_dir=Path(td) / "ng", log=lambda *a: None)
    s2.skip_grounding()
    s2.spec()
    s2.set_prompt_en(CT)
    assert s2.analysis()[0] is s2.analysis()[0], "analysis phải trả CÙNG một đối tượng"
    assert s2.analysis()[0].prompt_en == CT, s2.analysis()[0].prompt_en
    assert s2.genspec()[0].prompt_terms == [CT], s2.genspec()[0].prompt_terms
    print("--no-grounding: câu Culture-TRIP vẫn tới được bộ sinh")

print("ĐẠT")
