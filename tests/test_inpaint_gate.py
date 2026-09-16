"""Nấc inpaint chỉ được chọn cho bộ phận nhỏ; thiếu dáng tổng thể thì phải leo nấc khác."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ctig.agents.inpaint import locally_fixable, part_nouns_all
from ctig.agents import reflector as R
from ctig.schema import FilterVerdict

LOCAL = ["high stand-up mandarin collar", "wide obi sash tied at the back", "conical hat with a wide brim",
         "red silk sash at the waist", "round basket boat with a woven hull"]
WHOLE = ["fitted bodice with flowing loose panels", "worn over wide-legged long trousers",
         "long-sleeved tunic split at the hips into front and back panels"]
for a in LOCAL:
    assert locally_fixable(a), a
for a in WHOLE:
    assert not locally_fixable(a), a
# cụm ghép phải xét MỌI bộ phận, không chỉ cái dài nhất
assert part_nouns_all(WHOLE[2]) >= {"sleeve", "panels"}, part_nouns_all(WHOLE[2])
print("locally_fixable: %d bộ phận nhỏ nhận, %d phần thân áo từ chối" % (len(LOCAL), len(WHOLE)))


from ctig.schema import CulturalSpec, SpecEntity, GenSpec

SPEC = CulturalSpec(prompt_id="t", entities=[SpecEntity(
    entity_id="ao_dai", name_vi="áo dài", name_en="ao dai", required_attrs=["a"], forbidden_attrs=[],
    confusables=[], weight=1.0, required_attrs_en=LOCAL[:1] + WHOLE[:1], forbidden_attrs_en=[], kind="object")])


def verdict(missing):
    return FilterVerdict(path="/x.png", keep=True, matched_must_have=["a"], missing_must_have=list(missing),
                         matched_must_not=[], score=0.5)


for attr, want in ((LOCAL[0], "inpaint"), (WHOLE[0], None)):
    plan, tried, fix = R.decide(verdict([attr]), spec=SPEC, gen=GenSpec(prompt_id="t"), memory=[], patience=2,
                                name_en="ao dai", have_refs=True, log=lambda *a: None,
                                prior_fixes=[], have_inpaint=True)
    got = fix if fix == "inpaint" else None
    print("  thiếu '%s' -> nấc %s" % (attr[:34], fix))
    assert got == want, (attr, fix)
print("ĐẠT")
