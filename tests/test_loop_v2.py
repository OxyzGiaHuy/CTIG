"""Vòng lặp v2 (khung T2I-Copilot): không đọc KB, đích sửa lấy từ so với ảnh thật."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ctig.agents import copilot as C

# module này TUYỆT ĐỐI không được dính tới bảng kiểm viết tay.
# Bỏ chú thích và chuỗi trước khi soát, vì docstring có nhắc must_have để GIẢI THÍCH là không dùng.
import io
import tokenize

SRC = Path(__file__).resolve().parent.parent / "ctig" / "agents" / "copilot.py"
code = "".join(t.string + " " for t in tokenize.generate_tokens(io.StringIO(SRC.read_text()).readline)
               if t.type not in (tokenize.COMMENT, tokenize.STRING))
for bad in ("must_have", "must_not", "CulturalSpec", "required_attrs", "forbidden_attrs"):
    assert bad not in code, f"copilot.py còn dùng {bad} trong MÃ (không tính chú thích)"
# `schema` là biến cục bộ cho complete_json, nhưng KHÔNG được import ctig.schema hay kb
for line in SRC.read_text().splitlines():
    if line.startswith(("import ", "from ")):
        assert not any(m in line for m in ("schema", "kb", "knowledge")), f"không được import: {line}"
print("copilot.py: mã không đụng KB, must_have, must_not, CulturalSpec")

REPORT = {"subjects": ["a fisherman"], "attributes": ["round woven basket boat"], "spatial": [],
          "background": "beach at sunrise", "style": "photo", "entity_en": "Vietnamese thung chai basket boat",
          "look_alikes": ["a long wooden sampan", "a Chinese junk"]}


class FakeAgent:
    """Ảnh 'xau.png' sai thực thể và khác ảnh thật; 'dep.png' đúng."""
    class _LLM:
        def __init__(self, outer): self.o = outer
        def complete_json(self, system, user, schema, images=None, **kw):
            if "positive" in str(schema):     # LLM viết lại nhận xét thành mô tả đúng
                return {"positive": "a perfectly circular bowl-shaped hull woven from bamboo strips",
                        "negative": ["oval hull", "planked wood", "wooden sampan"]}
            bad = images and "xau" in images[0]
            if "differences" in str(schema):
                self.o.n_compare += 1
                return {"differences": ["hull is long and pointed, not round", "no woven bamboo texture"],
                        "foreign_elements": []} if bad else {"differences": [], "foreign_elements": []}
            v = 4 if bad else 9
            return {k: v for k in C.PROMPT_FIELDS + C.AESTHETIC_FIELDS}

    def __init__(self):
        self.llm = self._LLM(self)
        self.n_compare = 0

    def _complete(self, system, user, schema, max_new_tokens=None):
        return self.llm.complete_json(system, user, schema)

    def vqa_choice(self, q, img, n=3):
        p = 0.15 if "xau" in img else 0.92
        rest = (1 - p) / (n - 1)
        return [p] + [rest] * (n - 1)


ag = FakeAgent()
refs = ["/r1.jpg", "/r2.jpg", "/r3.jpg"]

bad = C.evaluate(ag, "/xau.png", REPORT, refs, log=lambda *a: None)
assert not bad.passed and bad.overall < 6, bad.overall
assert len(bad.differences) == 2 and bad.identity_p < 0.2
print("ảnh sai: %.1f/10, nhận nhầm là '%s', nêu %d khác biệt" % (bad.overall, bad.identity[:24], len(bad.differences)))

# nhận xét là chữ để ĐỌC; thứ đưa vào prompt phải là mô tả ĐÚNG, không chứa từ sai
tip = C.critique(bad, REPORT["entity_en"])
assert "hull is long" in tip, tip
pos, neg = C.suggestions(ag, bad, REPORT["entity_en"], log=lambda *a: None)
assert pos and "instead" not in pos.lower() and "not " not in pos.lower(), pos
for wrong in ("long", "pointed", "sampan"):
    assert wrong not in pos.lower(), f"câu thêm vào prompt còn chứa từ sai: {wrong} | {pos}"
assert neg, neg
print("nhận xét:", tip[:80])
print("thêm vào prompt:", pos[:80], "| negative:", neg)

good = C.evaluate(ag, "/dep.png", REPORT, refs, log=lambda *a: None)
assert good.passed and good.overall >= C.DEFAULT_THRESHOLD, good.overall
print("ảnh đúng: %.1f/10, đạt" % good.overall)

# vòng lặp: ảnh đầu xấu, vòng 1 sinh ra ảnh đẹp -> dừng ở vòng 1
seen = []
out = C.run_loop(ag, REPORT, "/xau.png", lambda pos, neg, n: (seen.append((n, pos, neg)) or "/dep.png"),
                 refs=refs, log=lambda *a: None)
assert out["final"] == "/dep.png" and len(out["rounds"]) == 1 and "đạt ở vòng 1" in out["stop"], out["stop"]
assert seen and "long" not in seen[0][1].lower(), seen[0][1]   # không được tuồn từ sai vào prompt
print("vòng lặp:", out["stop"])

# sinh lại luôn hỏng -> chạy hết 3 vòng, giữ ảnh tốt nhất
out2 = C.run_loop(ag, REPORT, "/xau.png", lambda pos, neg, n: "/xau2.png", refs=refs, log=lambda *a: None)
assert len(out2["rounds"]) == C.DEFAULT_MAX_ROUNDS and "hết 3 vòng" in out2["stop"], out2["stop"]
print("không cải thiện:", out2["stop"])

# không có ảnh thật -> vẫn chạy, trục văn hoá chỉ còn câu ép chọn
noref = C.evaluate(ag, "/xau.png", REPORT, [], log=lambda *a: None)
assert noref.axes["culture"] == 10.0 * noref.identity_p and any("không có ảnh thật" in n for n in noref.notes)
print("thiếu ảnh thật: không lỗi, trục văn hoá còn mỗi câu ép chọn")
print("ĐẠT")
