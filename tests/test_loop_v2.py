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
        self.llm.choice_prob = self._VL()
        self.n_compare = 0

    def _complete(self, system, user, schema, max_new_tokens=None):
        return self.llm.complete_json(system, user, schema)

    class _VL:
        """choice_prob cho câu đấu cặp: ảnh 'dep' luôn thắng ảnh 'xau'."""
        def __call__(self, q, images, letters):
            a, b = images[0], images[1]
            pa = 0.9 if ("dep" in a and "dep" not in b) else (0.1 if ("dep" in b and "dep" not in a) else 0.5)
            return [pa, 1 - pa]

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

# giữ mọi ảnh, ảnh cuối là ảnh ĐIỂM CAO NHẤT chứ không phải ảnh vòng cuối
seq = iter(["/xau2.png", "/dep.png", "/xau3.png"])
out3 = C.run_loop(ag, REPORT, "/xau.png", lambda pos, neg, n: next(seq), refs=refs, log=lambda *a: None)
assert set(out3["kept"]) == {"/xau.png", "/xau2.png", "/dep.png"}, out3["kept"]
assert out3["final"] == "/dep.png", "vòng 3 kém hơn thì phải giữ ảnh vòng 2"
print("giữ %d ảnh, vòng cuối kém -> vẫn lấy %s" % (len(out3["kept"]), out3["final"]))

# không có ảnh thật -> vẫn chạy, trục văn hoá chỉ còn câu ép chọn
noref = C.evaluate(ag, "/xau.png", REPORT, [], log=lambda *a: None)
assert noref.axes["culture"] == 10.0 * noref.identity_p and any("không có ảnh thật" in n for n in noref.notes)
print("thiếu ảnh thật: không lỗi, trục văn hoá còn mỗi câu ép chọn")
print("ĐẠT")

# --- hai cổng lọc thêm 2026-09-17, sau lượt chạy thật S001 với bộ chấm Mistral ------------------

# Lỗi thật: positive 'V-neck collar, long sleeves, fitted skirt, silk fabric' đi kèm negative
# ['wide collar', 'long sleeves', 'wide skirt', 'black book'] -> 'long sleeves' vừa bảo vẽ vừa cấm vẽ.
_neg = C._drop_contradictions("V-neck collar, long sleeves, fitted skirt, silk fabric",
                              ["wide collar", "long sleeves", "wide skirt", "black book"], log=lambda *a: None)
assert "long sleeves" not in _neg, _neg
assert _neg == ["wide collar", "wide skirt", "black book"], _neg
# không được cắt nhầm: 'short sleeves' ngược hẳn với 'long sleeves' nên phải giữ
_neg2 = C._drop_contradictions("Long silk dress with long sleeves, high collar",
                               ["short sleeves", "wide collar"], log=lambda *a: None)
assert _neg2 == ["short sleeves", "wide collar"], _neg2
print("cổng mâu thuẫn dương/âm: bỏ %d cụm, giữ %d" % (4 - len(_neg), len(_neg)))

# Lỗi thật: ô 'chi tiết văn hoá khác' trả về 'red flowers in bouquet' và 'black book' - đạo cụ trong cảnh,
# không phải dấu hiệu văn hoá, mà vẫn kéo trục văn hoá từ 10 xuống 0.
assert not C._names_a_culture("red flowers in bouquet")
assert not C._names_a_culture("black book")
assert not C._names_a_culture("large hoop earrings")
assert C._names_a_culture("Chinese floral brocade panel")
assert C._names_a_culture("cheongsam-style side fastening")
print("cổng 'phải nêu tên nền văn hoá': đạo cụ bị loại, hoa văn Trung Quốc được giữ")
print("ĐẠT (phần bổ sung)")

# --- bộ nhớ chống dao động, thêm 2026-09-17 sau lượt S001 -------------------------------------
_m = C.Memory()
_lg = lambda *a: None
assert _m.filter(["sleeves are too short", "collar is too wide"], _lg) == \
       ["sleeves are too short", "collar is too wide"]
_m.record("Long silk dress with long sleeves, high collar")
# vòng sau chê đúng thứ vừa yêu cầu -> bỏ, nếu không sẽ đi vòng tròn ngắn/dài/ngắn
assert _m.filter(["sleeves are too long", "collar is too wide"], _lg) == ["collar is too wide"]
# lời chê lặp tới lần thứ ba mà chưa sửa được -> thôi nhắc
assert _m.filter(["collar is too wide", "fabric is too thick"], _lg) == ["fabric is too thick"]
print("bộ nhớ: bỏ lời chê ngược, và bỏ lời chê lặp lần thứ 3")
print("ĐẠT (bộ nhớ)")

# --- trục văn hoá phải theo MỨC NGHIÊM TRỌNG, không theo SỐ lời chê (sửa 2026-09-17) ------------
# Lỗi thật trên S002: vòng 0 chê "thiếu đòn gánh" (khiếm khuyết định danh), vòng 1 sửa đúng rồi nhưng
# bộ chấm chuyển sang chê "thúng hơi nhỏ / chưa đủ sâu / chưa đủ tròn" — vẫn đủ 3 lời chê nên điểm văn
# hoá y nguyên 7,0. Công thức cũ 10−3·số_khác_biệt chỉ nhận được hai giá trị 7,0 và 8,0 trên thực tế.
class _FidAgent(FakeAgent):
    """Trả cùng SỐ lời chê cho hai ảnh, nhưng mức nghiêm trọng khác hẳn."""
    def __init__(self, fid):
        super().__init__()
        self._fid = fid

    class _LLM(FakeAgent._LLM):
        def complete_json(self, system, user, schema, images=None, **kw):
            if "differences" in str(schema) and "positive" not in str(schema):
                return {"differences": ["a", "b", "c"], "foreign_elements": [],
                        "cultural_fidelity": self.o._fid}
            return super().complete_json(system, user, schema, images=images, **kw)


def _culture(fid):
    ag = _FidAgent(fid)
    ag.llm = _FidAgent._LLM(ag)
    return C.evaluate(ag, "/dep.png", REPORT, ["/ref1.jpg", "/ref2.jpg"], log=lambda *a: None).axes["culture"]


_nghiem_trong = _culture(2)    # trông như vật của nền văn hoá khác
_nhe = _culture(9)             # đúng vật, khác vài chi tiết nhỏ
assert _nghiem_trong == 2.0 and _nhe == 9.0, (_nghiem_trong, _nhe)
assert _nhe - _nghiem_trong == 7.0, "cùng 3 lời chê mà mức nghiêm trọng khác thì điểm phải khác"
print("trục văn hoá: cùng 3 lời chê, fid 2 -> %.1f và fid 9 -> %.1f (cũ thì cả hai đều 7,0)"
      % (_nghiem_trong, _nhe))

# bộ nhớ: luật mới khớp 2 từ chung, bắt được ca đã lọt lưới
_m2 = C.Memory()
_m2.record("Large round woven baskets on a shoulder pole")
assert _m2.filter(["baskets are round instead of oval"], lambda *a: None) == []
print("bộ nhớ: bắt được 'baskets are round instead of oval' sau khi đã yêu cầu thúng tròn")
print("ĐẠT (trục văn hoá theo mức nghiêm trọng)")

# --- câu mô tả phải CỘNG DỒN qua các vòng (lỗi tìm ra 2026-09-17) -------------------------------
# run_loop_v2 chốt base_terms một lần rồi mỗi vòng chỉ nối câu của chính vòng đó, nên prompt vòng 3
# mất sạch phần vòng 1 và 2 đã sửa — trong khi negative lại cộng dồn. Bất đối xứng đó làm vòng lặp
# không tích luỹ: sửa xong tay áo ở vòng 1 thì vòng 2 quên, lỗi cũ quay lại.
_lgq = lambda *a: None
_ap = C.merge_positive([], "Long silk dress with long sleeves", log=_lgq)
_ap = C.merge_positive(_ap, "High mandarin collar", log=_lgq)
assert _ap == ["Long silk dress with long sleeves", "High mandarin collar"], _ap
# câu mới nói cùng chỗ (chung >= 2 từ) thì thay câu cũ, không chồng hai lệnh ngược nhau
_ap = C.merge_positive(_ap, "Fitted long sleeves, narrow cuffs", log=_lgq)
assert "Long silk dress with long sleeves" not in _ap and len(_ap) == 2, _ap
# giữ tối đa `keep` câu vì prompt Culture-TRIP đã 130-324 token CLIP
_ap = C.merge_positive(_ap, "White silk fabric", log=_lgq)
_ap = C.merge_positive(_ap, "Bare feet on wet sand", keep=3, log=_lgq)
assert len(_ap) == 3, _ap
print("prompt cộng dồn: giữ %d câu, câu nói cùng chỗ thì câu mới thắng" % len(_ap))
print("ĐẠT (cộng dồn câu mô tả)")

# --- ba lỗi soát mã phát hiện 2026-09-17 ------------------------------------------------------
# 1. Đường lùi của suggestions() đẩy nhận xét THÔ vào negative prompt: 'hull is oval instead of
#    circular' làm SDXL tránh vẽ 'circular' — cấm đúng thứ mình muốn. Đúng lỗi S012 quay lại cửa sau.
assert C._wrong_half("hull is oval instead of circular") == "hull is oval"
assert C._wrong_half("sides are planked wood instead of woven bamboo") == "sides are planked wood"
assert C._wrong_half("beef slices are thick, not thin") == "beef slices are thick"
assert C._wrong_half("collar is too wide") == "collar is too wide"   # không có 'instead of' thì giữ nguyên
print("negative không còn nuốt phần ĐÚNG của câu chê")

# 2. Bộ lọc phủ định so chuỗi con nên 'a kimono sleeve' bị loại vì trong 'kimono ' có 'no '.
_bad = ("not", "instead", "without", "no", "avoid", "remove", "avoiding", "removing")
for _s, _want in [("a kimono sleeve", False), ("no collar", True), ("a nostalgic scene", False),
                  ("not fitted", True), ("a notable pattern", False)]:
    _hit = bool((C._content_words(_s) | set(_s.lower().split())) & set(_bad))
    assert _hit == _want, (_s, _hit, _want)
print("bộ lọc phủ định so theo TỪ: 'kimono' và 'nostalgic' không còn bị loại oan")
print("ĐẠT (ba lỗi soát mã)")
