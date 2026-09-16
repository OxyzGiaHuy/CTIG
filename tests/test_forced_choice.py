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
print("ĐẠT")
