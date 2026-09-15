"""Agent bằng luật, offline. Dùng cho test và làm baseline. Không dịch được sang tiếng Anh."""

from __future__ import annotations

from ..kb import KnowledgeBase, contains
from ..schema import (
    Adjudication, AnalysisResult, Critique, CulturalSpec, GenSpec, Keyword,
    Perception, Prompt, RevisionPlan, SearchResult, SpecEntity,
)
from . import shared
from .rules import EXPANSION_RULES, REGION_LEXICON, SCENE_LEXICON


def _kb_en(vi_list: list[str], en_list: list[str], attr: str) -> str:
    """Bản tiếng Anh viết tay trong KB cho một thuộc tính tiếng Việt, hoặc "" nếu không có."""
    try:
        i = vi_list.index(attr)
    except ValueError:
        return ""
    return en_list[i] if i < len(en_list) else ""


class RuleAgent:
    name = "rule"

    def analyze(self, prompt: Prompt, kb: KnowledgeBase) -> AnalysisResult:
        text = f"{prompt.text_vi} {prompt.text_en}"
        kws: list[Keyword] = []
        region = None
        for reg, cues in REGION_LEXICON.items():
            if any(contains(text, c) for c in cues):
                region = reg
                kws.append(Keyword(reg, "region", "surface", 0.9, "khớp địa danh"))
                break
        surface: list[str] = []
        for ent, score, term in kb.match_text(text):
            surface.append(ent.id)
            kws.append(Keyword(ent.name_vi, "entity", "surface", score, f"prompt nhắc '{term}'"))
        for cue, label in SCENE_LEXICON.items():
            if contains(text, cue):
                kws.append(Keyword(label, "scene", "surface", 0.6, f"khớp '{cue}'"))
        expanded: list[str] = []
        for rule in EXPANSION_RULES:
            if rule.only_region is not None and region not in rule.only_region:
                continue
            fired = (rule.when_entity in surface) if rule.when_entity else False
            fired = fired or (bool(rule.when_text) and any(contains(text, c) for c in rule.when_text))
            if not fired:
                continue
            for eid in rule.add:
                if eid in surface or eid in expanded or kb.get(eid) is None:
                    continue
                expanded.append(eid)
                kws.append(Keyword(kb.get(eid).name_vi, "entity", "expanded", rule.confidence, rule.why))
        for eid in surface + expanded:
            for a in kb.get(eid).must_have[:2]:
                kws.append(Keyword(a, "attribute", "expanded", 0.8, f"thuộc tính của {kb.get(eid).name_vi}"))
        return AnalysisResult(prompt.id, kws, surface + expanded, region, prompt_en=prompt.text_en,
                              notes=None if (surface or expanded) else "Không nhận ra thực thể nào trong KB.")

    def build_spec(self, prompt, analysis, search, kb, max_entities, min_score) -> CulturalSpec:
        merged: dict[str, dict] = {}
        for it in search.items:
            if it.entity_id == "-":
                continue  # nhóm truy vấn "prompt gốc": chỉ để hiển thị, không vào spec
            s = merged.setdefault(it.entity_id, {"score": 0.0, "mh": [], "mn": [], "cf": [], "titles": [], "ref": None})
            s["score"] = max(s["score"], it.score)
            for a in it.must_have:
                a in s["mh"] or s["mh"].append(a)
            for a in it.must_not:
                a in s["mn"] or s["mn"].append(a)
            for c in it.confusable_with:
                c in s["cf"] or s["cf"].append(c)
            it.title in s["titles"] or s["titles"].append(it.title)
            if it.kind == "image" and it.local_path and it.is_reference:
                s["ref"] = it.local_path  # retrieval đã chọn một ảnh tốt nhất đạt ngưỡng, thực thể vật thể
        dropped: list[list[str]] = []
        kept = []
        from ..kb import contains

        ptext = f"{prompt.text_vi} {prompt.text_en}"
        for eid, s in merged.items():
            ent = kb.get(eid)
            if ent is None:
                dropped.append([eid, "không có trong KB"]); continue
            named_in_prompt = any(contains(ptext, term) for term in ent.search_terms if len(term) >= 3)
            if s["score"] < min_score and not named_in_prompt:
                dropped.append([eid, f"điểm {s['score']:.2f} < {min_score}"]); continue
            # Luật vùng chỉ áp cho thực thể SUY RA. v1.6 p012: "thuyền thúng" nêu tên thẳng mà bị bỏ vì Analysis đoán vùng bac_bo
            # trong khi KB ghi trung_bo -> spec rỗng, CLIP 0, không ảnh tham chiếu. Vùng do VLM đoán không được thắng chữ trong prompt.
            if analysis.region_hint and ent.region not in ("toan_quoc", analysis.region_hint) and not named_in_prompt:
                dropped.append([eid, f"vùng '{ent.region}' xung đột với '{analysis.region_hint}'"]); continue
            if not s["mh"]:
                dropped.append([eid, "không có thuộc tính kiểm chứng được"]); continue
            kept.append((eid, s))

        # Xếp hạng: nêu tên thẳng > thực thể bối cảnh có keyword (Tết không bị cắt) > còn lại theo điểm.
        named = {k.term for k in analysis.keywords if k.kind == "entity" and k.source == "surface"}
        mentioned = {k.term for k in analysis.keywords if k.kind == "entity"}

        def rank(p):
            ent = kb.get(p[0])
            if ent.name_vi in named:
                tier = 0
            elif ent.kind == "context" and ent.name_vi in mentioned:
                tier = 1
            else:
                tier = 2
            return (tier, -p[1]["score"])

        kept.sort(key=rank)
        for eid, _ in kept[max_entities:]:
            dropped.append([eid, f"vượt giới hạn {max_entities} thực thể"])
        kept = kept[:max_entities]
        weights = [1.0, 0.8, 0.65, 0.55]
        ents = []
        for i, (eid, s) in enumerate(kept):
            ent = kb.get(eid)
            # Bản tiếng Anh: lấy từ KB viết tay theo đúng vị trí; thuộc tính rút thêm (không có trong KB)
            # để "" -> PromptAgent dịch nốt, không dịch được thì bỏ khỏi prompt (không đưa tiếng Việt).
            mh_en = [_kb_en(ent.must_have, ent.must_have_en, a) for a in s["mh"]]
            mn_en = [_kb_en(ent.must_not, ent.must_not_en, a) for a in s["mn"]]
            ents.append(SpecEntity(eid, ent.name_vi, ent.name_en, s["mh"], s["mn"], s["cf"],
                                   weights[min(i, 3)], s["titles"],
                                   required_attrs_en=mh_en, forbidden_attrs_en=mn_en,
                                   clip_label=ent.clip_label or f"a photo of Vietnamese {ent.name_en.split('(')[0].strip()}",
                                   kind=ent.kind, reference_image=s["ref"],
                                   tags_en=list(ent.tags_en), neg_tags_en=list(ent.neg_tags_en)))
        return CulturalSpec(prompt.id, ents, [k.term for k in analysis.keywords if k.kind == "scene"], dropped)

    def extract_evidence(self, ent, texts):
        """Agent luật không đọc hiểu được văn bản. Trả rỗng để stage extraction bỏ qua."""
        return {"must_have": [], "must_not": [], "confusable_with": [], "attr_sources": {}}

    def critique(self, prompt, spec, perception) -> Critique:
        return shared.checklist_critique(spec, perception) or shared.rule_critique(spec, perception)

    def adjudicate(self, critique, perception, spec, threshold, clip_weight, drift_margin) -> Adjudication:
        return shared.adjudicate(critique, perception, spec, threshold, clip_weight, drift_margin)

    def plan_revision(self, adjudication, spec, gen_spec, kb, lora_available, reference_available) -> RevisionPlan:
        return shared.plan_revision(adjudication, spec, gen_spec, kb, lora_available, reference_available)

    # ------------------------------------------------------------ v1.4 agents (luật, offline)
    def summarize(self, se, ent, texts):
        mh = list(ent.must_have) if ent is not None else []
        return {"facts_vi": mh[:4], "facts_en": [a for a in (ent.must_have_en if ent is not None else []) if a][:4],
                "confusions_en": [f"unlike {c.get('name')}" for c in (ent.confusable_with if ent is not None else [])[:2]],
                "depiction_en": (ent.clip_label or "") if ent is not None else "",
                "dimensions": {"attire" if (ent is not None and ent.category == "trang_phuc") else "objects":
                               [a for a in (ent.must_have_en if ent is not None else []) if a][:4]}}

    def describe_image(self, path):
        return {"people_count": 1, "subjects": ["person"], "garments": ["fitted tunic with high stand-up collar over wide trousers"],
                "objects": [], "background": "stub card", "watermark_or_text": True}

    def match_descriptors(self, description, must_have_en, must_not_en):
        from ..kb import tokens

        td = tokens(description)
        have = [a for a in must_have_en if a and len(tokens(a) & td) / max(1, len(tokens(a))) >= 0.5]
        not_ = [a for a in must_not_en if a and len(tokens(a) & td) / max(1, len(tokens(a))) >= 0.6]
        return {"present_must_have": have, "present_must_not": not_, "unsure": []}

    def rank_candidates(self, prompt_en, brief_txt, items):
        order = sorted(items, key=lambda it: (len(it["must_not_seen"]), -len(it["must_have_seen"]), -it["metric_score"]))
        return {"order": [it["id"] for it in order], "reasons": {it["id"]: "luật: must_not, must_have, metric" for it in order}}

    def judge(self, prompt, spec, perception) -> tuple[float, str]:
        if not spec.entities:
            return 0.0, "Không có thực thể để đánh giá."
        ident, comp, foreign, notes = 0, [], 0, []
        for se in spec.entities:
            el, st = shared.locate(se, perception.elements)
            if st == "match":
                ident += 1
                from ..kb import attr_covered
                c = sum(1 for a in se.required_attrs if attr_covered(a, el.attrs))
                comp.append(c / max(1, len(se.required_attrs)))
                notes.append(f"{se.name_vi}: đúng, {c}/{len(se.required_attrs)} thuộc tính")
            elif st == "drift":
                foreign += 1; comp.append(0.0); notes.append(f"{se.name_vi}: lệch thành '{el.label}'")
            else:
                comp.append(0.0); notes.append(f"{se.name_vi}: vắng")
        n = len(spec.entities)
        score = (ident / n + sum(comp) / n + (1 - foreign / n)) / 3
        return score, "; ".join(notes)
