"""The three SAVIER agents. One MLLM backbone, three system prompts.

    C  Source-Aware Prompt Curator      text          prompt + curated Wikipedia -> Preservation Card + Evidence Card
    O  Prompt-Blind Image Observer      vision        the image ONLY (no prompt, no sources) -> exhaustive report
    R  Discrepancy-Guided Prompt Refiner text          cards + report -> gap analysis, <=3 positive repair actions,
                                                       phrases to drop from the expansion text

Every model output passes a MACHINE check before it is used; the checks encode failure modes that were
observed, not hypothesised (see the comments next to each one).
"""
from __future__ import annotations

import json
import re
import time

from .wiki import cut_passage

NEGATION = {"no", "not", "without", "never", "avoid", "don't", "dont", "instead", "remove", "none"}
FRAMING = ("close-up", "close up", "wide shot", "full body", "full-body", "top-down", "top down", "camera angle",
           "zoom", "crop", "lighting", "background", "composition", "framing", "portrait shot")


def _arr(): return {"type": "array", "items": {"type": "string"}}
def _norm(s): return re.sub(r"\s+", " ", str(s or "")).strip().lower()


class Transcript:
    """Every agent call, verbatim: system, user, images, JSON reply. This is the communication trace."""

    def __init__(self): self.rows: list[dict] = []

    def log(self, pid, agent, step, system, user, reply, images=None, seconds=0.0):
        self.rows.append({"t": time.strftime("%H:%M:%S"), "prompt_id": pid, "agent": agent, "step": step,
                          "system": system, "user": user, "images": images or [], "reply": reply, "seconds": round(seconds, 1)})


def _call(llm, tr, pid, agent, step, system, user, schema, images=None, max_new_tokens=900):
    t0 = time.time()
    try:
        d = llm.complete_json(system, user, schema, images=images, max_new_tokens=max_new_tokens) or {}
    except Exception as exc:  # noqa: BLE001
        d = {"_error": f"{type(exc).__name__}: {exc}"}
    tr.log(pid, agent, step, system, user, d, images, time.time() - t0)
    return d


# ====================================================================== C · Source-Aware Prompt Curator
CURATOR_SYSTEM = (
    "You are the Prompt & Cultural Analyst for a Vietnamese cultural text-to-image system.\n"
    "You produce TWO cards as one JSON object.\n\n"
    "CARD 1 — prompt_preservation: everything the prompt itself asks for, grouped. Take it ONLY from the "
    "prompt. Keep meaning intact. This card has the highest priority downstream. Needs no citation.\n\n"
    "CARD 2 — cultural_evidence: ONLY culturally identifying features that a camera can capture, taken ONLY "
    "from the Wikipedia passages given to you. Rules:\n"
    "- at most 3 identity_cues, at most 2 conditional_cues, at most 2 positive_disambiguators;\n"
    "- every identity_cue and conditional_cue must carry quote_vi: a VERBATIM sentence or phrase copied "
    "exactly from the Vietnamese passage, and source_url from the passage header;\n"
    "- if the passage does not clearly state a feature, do not invent it — list what you looked for under "
    "insufficient_evidence;\n"
    "- history, etymology, symbolism, taste, smell, sound go to discarded_nonvisual_facts;\n"
    "- a positive_disambiguator describes what the CORRECT object looks like where a look-alike would differ. "
    "Never describe the look-alike itself.\n"
    "- Never turn a prompt-specific detail (e.g. 'white') into a universal cultural feature.\n"
    "- Cues must describe the MAIN OBJECT's own appearance, or HOW it sits on the body (resting on one "
    "shoulder, worn over trousers, tied at the waist) when the source says so. Utensils, containers, "
    "serving temperature and preparation steps are not cues.\n"
    "Answer in JSON only."
)
CURATOR_A_SYSTEM = ("You are the Prompt Analyst. From the prompt ONLY, produce prompt_preservation: everything the prompt "
                    "asks for, grouped. Keep meaning intact; the English wording in must_preserve_verbatim must be copied "
                    "exactly. Needs no citation. JSON only.")
_CUE = {"type": "object", "properties": {"cue_vi": {"type": "string"}, "cue_en": {"type": "string"},
        "visual_check_vi": {"type": "string"}, "quote_vi": {"type": "string"}, "source_url": {"type": "string"}}}
PRESERVATION_SCHEMA = {"type": "object", "properties": {"prompt_preservation": {"type": "object", "properties": {
    "main_entity": {"type": "object", "properties": {"vi": {"type": "string"}, "en": {"type": "string"}}},
    "subject_attributes": _arr(), "supporting_objects": _arr(), "background_and_scene": _arr(),
    "actions_and_relations": _arr(), "colors": _arr(), "visual_effects": _arr(), "time_weather_lighting": _arr(),
    "counts": {"type": "object"}, "text_in_image": _arr(), "must_preserve_verbatim": _arr()}}}, "required": ["prompt_preservation"]}
EVIDENCE_SCHEMA = {"type": "object", "properties": {"cultural_evidence": {"type": "object", "properties": {
    "identity_cues": {"type": "array", "items": _CUE}, "conditional_cues": {"type": "array", "items": _CUE},
    "positive_disambiguators": {"type": "array", "items": {"type": "object", "properties": {
        "target_appearance_vi": {"type": "string"}, "target_appearance_en": {"type": "string"}, "source_url": {"type": "string"}}}},
    "discarded_nonvisual_facts": _arr(), "insufficient_evidence": _arr()}}}, "required": ["cultural_evidence"]}


def curator(llm, tr, pid, prompt_vi, prompt_en, pages, keywords, log=print) -> dict:
    """Two calls: the Preservation Card (prompt only) and the Evidence Card (curated Wikipedia only)."""
    a = _call(llm, tr, pid, "C", "preservation_card", CURATOR_A_SYSTEM,
              f"PROMPT VI: {prompt_vi}\nPROMPT EN: {prompt_en}\n\nReturn prompt_preservation as JSON.", PRESERVATION_SCHEMA, max_new_tokens=700)
    budget, parts = 4000, []                       # 4k chars TOTAL across pages, Vietnamese first
    for pg in sorted(pages, key=lambda x: 0 if x["lang"] == "vi" else 1):
        if budget <= 600: break
        seg = cut_passage(pg["text"], keywords, budget)
        parts.append(f"=== PASSAGE [{pg['lang']}] {pg['title']} — {pg['url']} ===\n{seg}"); budget -= len(seg)
    b = _call(llm, tr, pid, "C", "evidence_card", CURATOR_SYSTEM,
              f"PROMPT VI: {prompt_vi}\nPROMPT EN: {prompt_en}\n\nWIKIPEDIA PASSAGES (the only allowed source for cultural cues):\n"
              + "\n\n".join(parts) + "\n\nReturn cultural_evidence as JSON. quote_vi must be copied verbatim.", EVIDENCE_SCHEMA, max_new_tokens=2200)
    cards = {"prompt_preservation": a.get("prompt_preservation") or {}, "cultural_evidence": b.get("cultural_evidence") or {}}
    # ---- machine check: quote must be verbatim in a Vietnamese source AND must actually be about the cue.
    # (A verbatim quote about boiling broth was once attached to a cue "stone bowl"; verbatim-ness alone is not entailment.)
    vi_text = _norm(" ".join(p["text"] for p in pages if p["lang"] == "vi")); urls = {p["url"] for p in pages}
    ce, dropped = cards["cultural_evidence"], []
    for key, cap in (("identity_cues", 3), ("conditional_cues", 2)):
        keep = []
        for c in ce.get(key) or []:
            q = _norm(c.get("quote_vi")); cue_words = {w for w in _norm(c.get("cue_vi")).split() if len(w) > 2}
            ok = len(q) >= 12 and q in vi_text and c.get("source_url") in urls and bool(cue_words & set(q.split()))
            (keep if ok else dropped).append(c if ok else f"{key}: '{str(c.get('cue_vi'))[:50]}' — quote not verbatim / not about the cue / foreign URL")
        ce[key] = keep[:cap]
    ce["positive_disambiguators"] = (ce.get("positive_disambiguators") or [])[:2]
    ce["insufficient_evidence"] = (ce.get("insufficient_evidence") or []) + dropped
    tr.log(pid, "C", "machine_check", "(quote_vi verbatim in vi source; cue words in quote; caps 3/2/2)", "", {"dropped": dropped})
    log(f"  [C] identity {len(ce['identity_cues'])} · conditional {len(ce['conditional_cues'])} · disambiguators {len(ce['positive_disambiguators'])} · dropped {len(dropped)}")
    return cards


# ====================================================================== O · Prompt-Blind Image Observer
OBSERVER_SYSTEM = (
    "You are a Blind Visual Observer. You receive ONE photograph and nothing else. You do not know what it "
    "was supposed to show.\n"
    "Report EXHAUSTIVELY what is visible, in this order: 1 main subjects; 2 clothing and appearance; "
    "3 foreground objects; 4 supporting objects; 5 background; 6 actions and spatial relations; 7 colours; "
    "8 counts; 9 small effects such as smoke, steam, light, motion, reflections; 10 uncertain regions.\n"
    "RULES: describe only what a camera captured — shapes, materials, how parts join, counts, colours. "
    "Never say whether anything is correct, authentic or traditional. Never name a country, culture or "
    "ethnicity. Never guess at anything outside the frame; list cut-off parts under uncertain. "
    "Short noun phrases, 3-12 words each. JSON only."
)
OBSERVER_SCHEMA = {"type": "object", "properties": {
    "main_subjects": _arr(), "clothing_and_appearance": _arr(), "foreground_objects": _arr(), "supporting_objects": _arr(),
    "background": _arr(), "actions": _arr(), "spatial_relations": _arr(), "colors": _arr(), "counts": {"type": "object"},
    "visible_effects": _arr(), "uncertain": _arr()}, "required": ["main_subjects"]}


def observer(llm, tr, pid, image, step, log=print) -> dict:
    d = _call(llm, tr, pid, "O", step, OBSERVER_SYSTEM, "Describe this photograph. Return JSON.", OBSERVER_SCHEMA, images=[image])
    log(f"  [O·{step}] {sum(len(v) for v in d.values() if isinstance(v, list))} observations")
    return d


CHECK_SYSTEM = ("You rate how well ONE photograph matches ONE written statement. score 0-10: 10 exactly as described, "
                "5 the thing is there but clearly differs, 0 nothing of the kind. If the part lies outside the picture set "
                "in_frame false and give no score. Never name a country or culture. evidence: what you see, max 15 words. JSON only.")
CHECK_SCHEMA = {"type": "object", "properties": {"in_frame": {"type": "boolean"}, "score": {"type": "number"}, "evidence": {"type": "string"}},
                "required": ["in_frame"]}


def check_statement(llm, tr, pid, image, statement, step):
    """Targeted single-image question, 0-10. Used by the Refiner to drop actions already satisfied in I0."""
    d = _call(llm, tr, pid, "O", step, CHECK_SYSTEM, f'Statement: "{statement}"\nHow fully does this photograph match that statement?\n'
              'Return JSON: {"in_frame": true|false, "score": 0-10, "evidence": ".."}', CHECK_SCHEMA, images=[image], max_new_tokens=120)
    try:
        return (float(d.get("score")) if d.get("in_frame", True) else None), str(d.get("evidence") or "")[:100]
    except (TypeError, ValueError):
        return None, ""


# ====================================================================== R · Discrepancy-Guided Prompt Refiner
REFINER_SYSTEM = (
    "You are the Gap Analyzer & Prompt Refiner. You compare a blind visual report of a generated image "
    "against (a) the Prompt Preservation Card and (b) the Cultural Evidence Card.\n"
    "Classify: missing_prompt_explicit (asked by the prompt, not in the report); missing_cultural_identity "
    "(identity cue not evidenced in the report); contradictions (report shows something incompatible); "
    "already_satisfied; uncertain_no_repair (the report is unsure — do NOT repair these).\n"
    "Priority for repair: 1 details stated in the prompt; 2 supporting objects, background, relations; "
    "3 identity cues; 4 conditional cues. At most 3 repair_actions.\n"
    "Each repair_action is ONE positive English instruction (max 18 words) describing exactly what should be "
    "visibly present, including WHERE on the body or scene it sits when that matters (e.g. 'a bamboo pole "
    "resting across one shoulder with a basket hanging from each end'). Never use negation (no/not/without/avoid). "
    "Never name the wrong object. Never change framing, camera angle, lighting or composition. Do not restate "
    "things already satisfied.\n"
    "drop_phrases: copy VERBATIM any phrase from the EXPANSION TEXT (never from the original prompt) that "
    "contradicts the Cultural Evidence Card or pushes toward a look-alike object. Empty list if none. JSON only."
)
REFINER_SCHEMA = {"type": "object", "properties": {
    "missing_prompt_explicit": _arr(), "missing_cultural_identity": _arr(), "contradictions": _arr(), "already_satisfied": _arr(),
    "uncertain_no_repair": _arr(), "repair_actions": _arr(), "drop_phrases": _arr()}, "required": ["repair_actions"]}


def _clean_action(a: str) -> str | None:
    low = " " + _norm(a) + " "
    return None if (set(low.split()) & NEGATION or any(k in low for k in FRAMING)) else " ".join(str(a).split())


def refiner(llm, tr, pid, prompt_en, expansion, cards, report, i0, log=print) -> dict:
    d = _call(llm, tr, pid, "R", "gap_analysis", REFINER_SYSTEM,
              f"ORIGINAL PROMPT: {prompt_en}\n\nEXPANSION TEXT (added by Culture-TRIP, may be trimmed):\n{expansion}\n\n"
              f"PROMPT PRESERVATION CARD:\n{json.dumps(cards['prompt_preservation'], ensure_ascii=False)}\n\nCULTURAL EVIDENCE CARD:\n"
              f"{json.dumps(cards['cultural_evidence'], ensure_ascii=False)}\n\nBLIND VISUAL REPORT OF THE IMAGE:\n"
              f"{json.dumps(report, ensure_ascii=False)}\n\nReturn the gap analysis as JSON.", REFINER_SCHEMA)
    raw = [str(x) for x in d.get("repair_actions") or []]
    actions = [y for y in (_clean_action(x) for x in raw) if y]
    # ---- machine check 1: drop_phrases must occur verbatim in the expansion and never in the original prompt.
    drop = [" ".join(str(x).split()) for x in d.get("drop_phrases") or []]
    drop = [x for x in drop if len(x) >= 6 and x.lower() in expansion.lower() and x.lower() not in prompt_en.lower()][:3]
    # ---- machine check 2: score each action on I0 first; an action already >= 8 repairs nothing (a "high collar"
    # action once took one of three slots on an image that already had one). Fill free slots with identity cues.
    keep, already = [], []
    for act in actions:
        sc, _ = check_statement(llm, tr, pid, i0, act, "pre_score_I0")
        (already if (sc is not None and sc >= 8) else keep).append((act, sc))
    if len(keep) < 3:
        for c in cards["cultural_evidence"].get("identity_cues") or []:
            cue = str(c.get("cue_en") or "").strip()
            if cue and all(cue.lower() not in a.lower() for a, _ in keep):
                sc, _ = check_statement(llm, tr, pid, i0, cue, "pre_score_I0")
                if sc is None or sc < 8:
                    keep.append((f"clearly visible: {cue}", sc))
            if len(keep) >= 3: break
    d["repair_actions"], d["drop_phrases"] = [a for a, _ in keep][:3], drop
    tr.log(pid, "R", "machine_check", "(negation/framing filter; drop_phrases verbatim in expansion; actions already >=8 on I0 removed)", "",
           {"kept": keep, "already_satisfied_on_I0": already, "drop_phrases": drop})
    log(f"  [R] missing-prompt {len(d.get('missing_prompt_explicit') or [])} · missing-culture {len(d.get('missing_cultural_identity') or [])} · "
        f"actions {len(d['repair_actions'])} · drop {len(drop)}" + (f" · removed {len(already)} already-satisfied" if already else ""))
    return d


def build_P1(p_ct: str, p0: str, actions: list[str], card: dict, drop: list[str]) -> str:
    """P1 = P0 verbatim + (expansion minus dropped phrases) + preserve clause naming the card's supporting
    objects + <=3 positive repair actions. No negative prompt anywhere."""
    if not actions:
        return p_ct
    if p_ct.startswith(p0) and drop:
        exp = p_ct[len(p0):]
        for x in drop:
            i = exp.lower().find(x.lower())
            if i >= 0: exp = exp[:i] + exp[i + len(x):]
        p_ct = p0 + " " + " ".join(exp.replace(" ,", ",").replace(" .", ".").split())
    keep = [str(x).strip() for k in ("supporting_objects", "background_and_scene") for x in (card.get(k) or [])
            if re.search(r"[A-Za-z]", str(x)) and not re.search(r"[ăâđêôơưàáảãạ]", str(x).lower())]
    keep_clause = f" Keep clearly visible: {'; '.join(dict.fromkeys(keep))}." if keep else ""
    acts = "\n".join(f"{i + 1}. {a.rstrip('.')}." for i, a in enumerate(actions))
    return (f"{p_ct.strip()}\n\nPreserve the current subject, composition, setting, colors, and all correctly rendered details."
            f"{keep_clause} Make these additions clearly visible:\n{acts}")
