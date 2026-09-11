"""Agent bằng luật, offline. Dùng cho test và làm baseline. Không dịch được sang tiếng Anh."""

from __future__ import annotations

from ..kb import KnowledgeBase, contains
from ..schema import (
    Adjudication, AnalysisResult, Critique, CulturalSpec, GenSpec, Keyword,
    Perception, Prompt, RevisionPlan, SearchResult, SpecEntity,
)
from . import shared
from .rules import EXPANSION_RULES, REGION_LEXICON, SCENE_LEXICON


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
            s = merged.setdefault(it.entity_id, {"score": 0.0, "mh": [], "mn": [], "cf": [], "titles": [], "ref": None})
            s["score"] = max(s["score"], it.score)
            for a in it.must_have:
                a in s["mh"] or s["mh"].append(a)
            for a in it.must_not:
                a in s["mn"] or s["mn"].append(a)
            for c in it.confusable_with:
                c in s["cf"] or s["cf"].append(c)
            it.title in s["titles"] or s["titles"].append(it.title)
            if it.kind == "image" and it.local_path and (it.clip_match or 0) >= 0.5:
                s["ref"] = it.local_path
        dropped: list[list[str]] = []
        kept = []
        for eid, s in merged.items():
            ent = kb.get(eid)
            if ent is None:
                dropped.append([eid, "không có trong KB"]); continue
            if s["score"] < min_score:
                dropped.append([eid, f"điểm {s['score']:.2f} < {min_score}"]); continue
            if analysis.region_hint and ent.region not in ("toan_quoc", analysis.region_hint):
                dropped.append([eid, f"vùng '{ent.region}' xung đột với '{analysis.region_hint}'"]); continue
            if not s["mh"]:
                dropped.append([eid, "không có thuộc tính kiểm chứng được"]); continue
            kept.append((eid, s))
        named = {k.term for k in analysis.keywords if k.kind == "entity" and k.source == "surface"}
        kept.sort(key=lambda p: (0 if kb.get(p[0]).name_vi in named else 1, -p[1]["score"]))
        for eid, _ in kept[max_entities:]:
            dropped.append([eid, f"vượt giới hạn {max_entities} thực thể"])
        kept = kept[:max_entities]
        weights = [1.0, 0.8, 0.65, 0.55]
        ents = []
        for i, (eid, s) in enumerate(kept):
            ent = kb.get(eid)
            ents.append(SpecEntity(eid, ent.name_vi, ent.name_en, s["mh"], s["mn"], s["cf"],
                                   weights[min(i, 3)], s["titles"],
                                   required_attrs_en=list(s["mh"]), forbidden_attrs_en=list(s["mn"]),
                                   reference_image=s["ref"]))
        return CulturalSpec(prompt.id, ents, [k.term for k in analysis.keywords if k.kind == "scene"], dropped)

    def critique(self, prompt, spec, perception) -> Critique:
        return shared.rule_critique(spec, perception)

    def adjudicate(self, critique, perception, spec, threshold, clip_weight, drift_margin) -> Adjudication:
        return shared.adjudicate(critique, perception, spec, threshold, clip_weight, drift_margin)

    def plan_revision(self, adjudication, spec, gen_spec, kb, lora_available, reference_available) -> RevisionPlan:
        return shared.plan_revision(adjudication, spec, gen_spec, kb, lora_available, reference_available)

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
