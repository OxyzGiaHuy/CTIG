"""Kiểm nhánh trắc nghiệm hai lựa chọn của Reviewer bằng agent giả (không cần torch/PIL)."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from ctig.agents import describe as D
from ctig.schema import CulturalSpec, SpecEntity, ImageDescriptor, FilterVerdict, to_dict, from_dict

HAVE = "long-sleeved tunic split at the hips into front and back panels"
ALT = "a one-piece dress joined from top to bottom, no separate panels"


class FakeAgent:
    """Gật mọi câu có/không (mô phỏng thiên lệch thật), nhưng câu trắc nghiệm chọn đúng."""
    def __init__(self, jumpsuit): self.jumpsuit, self.n_choice = jumpsuit, 0
    def match_descriptors(self, text, have, notd): return {"present_must_have": [], "present_must_not": []}
    def vqa_yes(self, q, img): return 0.87
    def attr_alternatives(self, name, attrs, notd): return {HAVE: ALT}
    def vqa_choice(self, q, img, n=3):
        self.n_choice += 1
        line_a = [l for l in q.splitlines() if l.startswith("A. ")][0][3:].lower()
        a_is_have = line_a.startswith(HAVE[:20].lower())
        p_have = 0.1 if self.jumpsuit else 0.9          # xác suất model chọn mô tả ĐÚNG
        pa = p_have if a_is_have else 1.0 - p_have
        return [pa * 0.98, (1 - pa) * 0.98, 0.02]


spec = CulturalSpec(prompt_id="t", entities=[SpecEntity(
    entity_id="ao_dai", name_vi="áo dài", name_en="ao dai", required_attrs=["x"], forbidden_attrs=[],
    confusables=[], weight=1.0, required_attrs_en=[HAVE], forbidden_attrs_en=[], kind="object")])
desc = ImageDescriptor(path="/x.png", people_count=1, garments=["a white outfit"])

D.FORCED_CHOICE = True          # tắt mặc định (xem ghi chú trong describe.py), bật để kiểm mã
for jumpsuit, want in ((True, "bác"), (False, "nhận")):
    ag = FakeAgent(jumpsuit)
    alts = D._alternatives(ag, spec, log=lambda *a: None)
    assert alts == {HAVE: ALT}, alts
    v = D._verdict(ag, desc, spec, "candidate", 1, clip=None, alts=alts)
    assert ag.n_choice == 2, f"phải hỏi cả hai thứ tự, hỏi {ag.n_choice}"
    got = "nhận" if HAVE in v.matched_must_have else "bác"
    print(f"jumpsuit={jumpsuit}: score {v.score:+.2f} vqa={v.vqa.get(HAVE)} fc={v.vqa_fc.get(HAVE)} -> {got} (cần {want})")
    assert got == want, f"cần {want}, được {got}"
    assert v.alt_attrs.get(HAVE) == ALT
    rt = from_dict(FilterVerdict, to_dict(v))
    assert rt.vqa_fc == v.vqa_fc and rt.alt_attrs == v.alt_attrs, "vòng JSON mất trường mới"

# không có backend trắc nghiệm -> quay về câu phủ định, không vỡ
class OldAgent(FakeAgent):
    vqa_choice = None
    def __init__(self): super().__init__(True)
ag = OldAgent()
assert D._alternatives(ag, spec, log=lambda *a: None) == {}
v = D._verdict(ag, desc, spec, "candidate", 1, clip=None, alts={})
print("không có vqa_choice: score %+.2f, fc rỗng=%s" % (v.score, not v.vqa_fc))
# ghép bằng luật: must_not cùng vùng được chọn làm mô tả sai, không cần hỏi LLM
HAVE4 = ["long-sleeved tunic split at the hips into front and back panels", "high stand-up mandarin collar",
         "worn over wide-legged long trousers", "fitted bodice with flowing loose panels"]
NOT4 = ["wide obi sash tied at the back", "one-piece dress with no trousers underneath",
        "diagonal Y-shaped crossed collar", "puffy flared skirt"]
pairs = D.pair_distractors(HAVE4, NOT4)
assert pairs[HAVE4[0]] == NOT4[1], pairs[HAVE4[0]]
assert pairs[HAVE4[1]] == NOT4[2], pairs[HAVE4[1]]
assert pairs[HAVE4[2]] == NOT4[3], pairs[HAVE4[2]]   # váy xoè hẹp hơn "áo liền quần" -> hợp với quần hơn
assert len(pairs) == 4, pairs
print("ghép luật: 4/4 thuộc tính có mô tả sai đúng vùng, không gọi LLM")


# LLM viết lệch vùng (lật tay áo thay vì lật phần tà) -> bị bỏ
class DriftAgent(FakeAgent):
    def __init__(self): super().__init__(True)
    def attr_alternatives(self, name, attrs, notd): return {HAVE: "short-sleeved tunic without any splits"}
    def vqa_choice(self, q, img, n=3): raise AssertionError("không được hỏi khi mô tả sai bị bỏ")


msgs = []
assert D._alternatives(DriftAgent(), spec, log=msgs.append) == {}
assert any("phủ định" in m for m in msgs), msgs
assert not D.usable_distractor(HAVE, "short-sleeved tunic without any splits")
assert not D.usable_distractor(HAVE, "tunic with no split at the hips")
assert D.usable_distractor(HAVE, "a jumpsuit joined from shoulder to ankle")
assert D.usable_distractor(HAVE, NOT4[1])
print("mô tả sai chỉ phủ định chính thuộc tính: bị bỏ đúng")

D.FORCED_CHOICE = False

# --- hiệu chỉnh ngưỡng trên ảnh thật (v1.9.3) ---
COLLAR = "high stand-up mandarin collar"
TROUSERS = "worn over wide-legged long trousers"
spec3 = CulturalSpec(prompt_id="t", entities=[SpecEntity(
    entity_id="ao_dai", name_vi="áo dài", name_en="ao dai", required_attrs=["a", "b", "c"], forbidden_attrs=[],
    confusables=[], weight=1.0, required_attrs_en=[HAVE, COLLAR, TROUSERS], forbidden_attrs_en=[], kind="object")])

# số đo thật từ v191/S001 trên ba ảnh áo dài thật
REF_VALS = {HAVE: [0.96, 0.93, 0.95], COLLAR: [0.75, 0.75, 0.97], TROUSERS: [0.96, 0.00, 0.20]}


class CalAgent:
    """Trả số đo thật cho ảnh tham chiếu; với ứng viên trả giá trị đặt sẵn."""
    def __init__(self, cand): self.cand, self.seen = cand, []
    def match_descriptors(self, t, h, n): return {"present_must_have": [], "present_must_not": []}
    def vqa_yes(self, q, img):
        attr = next(a for a in REF_VALS if a[:24] in q)
        v = REF_VALS[attr][int(img[-5])] if img.startswith("/ref") else self.cand[attr]
        return round(1.0 - v, 3) if "does NOT have" in q else v   # câu phủ định đối chứng nhất quán


D._CAL_CACHE.clear()
refs = ["/ref0.jpg", "/ref1.jpg", "/ref2.jpg"]
cal = D.calibrate(CalAgent({}), spec3, refs, log=lambda *a: None)
assert cal[TROUSERS]["checkable"] is False, cal[TROUSERS]   # 0,96/0,00/0,20 -> tà che quần, không kiểm được
assert cal[TROUSERS]["ref_spread"] > 0.45, cal[TROUSERS]
# thuộc tính TB cao nhưng ba ảnh thật bất đồng nhiều cũng bị loại
D._CAL_CACHE.clear()
REF_VALS[COLLAR] = [0.99, 0.98, 0.40]
c2 = D.calibrate(CalAgent({}), spec3, refs, log=lambda *a: None)
assert c2[COLLAR]["checkable"] is False, c2[COLLAR]
REF_VALS[COLLAR] = [0.75, 0.75, 0.97]
D._CAL_CACHE.clear()
cal = D.calibrate(CalAgent({}), spec3, refs, log=lambda *a: None)
print("loại theo độ phân tán: TB %.2f nhưng chênh %.2f -> bỏ" % (0.79, 0.59))
assert cal[HAVE]["checkable"] and 0.85 < cal[HAVE]["thr"] < 0.90, cal[HAVE]
assert cal[COLLAR]["thr"] < cal[HAVE]["thr"], (cal[COLLAR], cal[HAVE])
print("hiệu chỉnh: %s ngưỡng %.2f | %s ngưỡng %.2f | %s BỎ (ảnh thật %.2f)"
      % (HAVE[:18], cal[HAVE]["thr"], COLLAR[:18], cal[COLLAR]["thr"], TROUSERS[:18], cal[TROUSERS]["ref_mean"]))

# ảnh áo liền quần thật của v191: tà 0,85 cổ 0,97 quần 0,94 -> trước đây +1,00
JUMP = {HAVE: 0.85, COLLAR: 0.97, TROUSERS: 0.94}
GOOD = {HAVE: 0.96, COLLAR: 0.90, TROUSERS: 0.25}
got = {}
for lab, vals, want_have in (("áo liền quần", JUMP, False), ("áo dài đúng", GOOD, True)):
    D._CAL_CACHE.clear()
    ag2 = CalAgent(vals)
    cal2 = D.calibrate(ag2, spec3, refs, log=lambda *a: None)
    v = D._verdict(ag2, desc, spec3, "candidate", 1, clip=None, alts={}, calib=cal2)
    print("  %-14s điểm %+.2f · tà %s · bỏ khỏi bảng kiểm %s"
          % (lab, v.score, "có" if HAVE in v.matched_must_have else "THIẾU", v.unverifiable))
    assert (HAVE in v.matched_must_have) is want_have, (lab, v.matched_must_have)
    assert v.unverifiable == [TROUSERS], v.unverifiable
    assert TROUSERS not in v.missing_must_have
    got[lab] = v.score
assert got["áo dài đúng"] - got["áo liền quần"] >= 0.20, got
assert D._verdict(CalAgent(JUMP), desc, spec3, "candidate", 1, clip=None, alts={}, calib={}).score == 1.0, \
    "không hiệu chỉnh thì vẫn +1.00 như cũ"
# mọi thuộc tính đều không kiểm được -> KHÔNG bỏ hết, nếu không Reviewer mù
D._CAL_CACHE.clear()
allbad = D._verdict(CalAgent({a: 0.9 for a in REF_VALS}), desc, spec3, "candidate", 1, clip=None, alts={},
                    calib={a: {"side": "have", "checkable": False, "thr": 0.6} for a in REF_VALS})
assert allbad.unverifiable == [] and allbad.score > 0, (allbad.unverifiable, allbad.score)
print("mọi thuộc tính không kiểm được: giữ nguyên bảng kiểm thay vì chấm mù")
print("hiệu chỉnh ngưỡng: áo liền quần bị bác, áo dài đúng được nhận")
print("ĐẠT")
