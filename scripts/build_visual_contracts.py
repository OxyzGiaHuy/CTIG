#!/usr/bin/env python3
"""Build the paper-facing Visual Contract v2 file from the verified contracts.

The transformation is deterministic.  It does not invent cultural facts: factual
descriptions and Wikipedia sources are copied from ``ctig_contracts_verified.json``.
The additions are operational metadata used by the agents (visibility, scoring,
priority, prompt role, and safe generation cues).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

# Facts that are real but cannot reliably be judged in the requested view.
OPTIONAL = {
    ("S010", "underwater_control"),
    ("S027", "pork_knuckle_optional"),
}

CONDITIONAL = {
    ("S020", "inner_head_ring"),       # normally hidden once the hat is worn
    ("S026", "rice_paper_wrapper"),    # wrapper material is hard to infer after frying
    ("S029", "flattened_green_rice"),  # hidden when the requested parcel is closed
    ("S030", "long_rectangular_scarf"),# full shape is hidden while worn
    ("S036", "betel_areca_offering"),  # may be under the requested red tray cloth
    ("S043", "multiple_colour_blocks"),# not every block need be in the camera view
}

PROMPT_ACTION = {
    ("S002", "mobile_vendor_with_goods"),
    ("S009", "plucked_with_plectrum"),
    ("S012", "person_paddling"),
    ("S017", "male_female_responsive_singing"),
    ("S021", "children_lantern_procession"),
    ("S024", "coffee_dripping"),
    ("S035", "ritual_text_reading"),
    ("S043", "woodblock_printing_action"),
    ("S048", "hands_shape_wet_clay"),
    ("S050", "adult_gives_child"),
}

PROMPT_CONTEXT = {
    ("S014", "flooded_reflective_paddies"),
    ("S023", "kumquat_and_red_envelopes"),
    ("S024", "condensed_milk_and_ice"),
    ("S025", "peanut_hoisin_dip"),
    ("S028", "wheel_shaped_slices"),
    ("S029", "lotus_leaf_wrapper"),
    ("S029", "straw_tie"),
    ("S034", "courtyard_and_banyan"),
    ("S035", "altar_context"),
    ("S035", "ao_the_khan_xep_prompt_attire"),
    ("S041", "peach_blossoms_for_sale"),
    ("S044", "cut_woody_branch_in_vase"),
    ("S044", "tet_context"),
    ("S050", "new_year_gift_context"),
}

# Secondary evidence helps recognition but should not outrank silhouette/structure.
SECONDARY = {
    ("S003", "onion_garnish"),
    ("S004", "bamboo_strip_ties"),
    ("S008", "lantern_lit_night_street"),
    ("S017", "male_traditional_attire"),
    ("S017", "female_layered_attire"),
    ("S017", "female_hat_and_headscarf"),
    ("S023", "tet_food_marker"),
    ("S025", "peanut_hoisin_dip"),
    ("S026", "herbs_and_fish_sauce_dip"),
    ("S033", "formal_court_setting"),
    ("S034", "courtyard_and_banyan"),
    ("S037", "coffee_service"),
    ("S040", "chin_strap"),
    ("S041", "tet_goods_and_shoppers"),
    ("S044", "tet_context"),
    ("S045", "five_fruit_tray"),
    ("S050", "new_year_gift_context"),
}

# Neutral camera regions for Observer attention.  These name a part to inspect,
# not the culturally correct value of that part, so they do not leak the answer.
FOCUS = {
    "S002": ["carrying pole", "loads", "vendor and goods"],
    "S005": ["bread", "filling", "filling"],
    "S007": ["pavilion", "stelae", "stele supports"],
    "S011": ["performers", "performance", "stage"],
    "S014": ["terrain", "terrace boundaries", "water surfaces"],
    "S016": ["roof", "building supports", "surroundings"],
    "S019": ["headwear", "outer garment", "garment decoration"],
    "S021": ["people and action", "lantern shape", "lantern supports"],
    "S022": ["lion costume", "costume operators", "accompanying character"],
    "S023": ["meal", "food", "decorations and gifts"],
    "S024": ["filter and glass", "liquid flow", "drink contents"],
    "S025": ["wrapper", "filling", "dipping sauce"],
    "S026": ["rolls", "wrapper", "garnish and dipping sauce"],
    "S027": ["noodles", "broth and meat", "toppings"],
    "S028": ["whole cake", "cut pieces", "cut surface"],
    "S029": ["rice", "wrapper", "tie"],
    "S032": ["instrument body", "strings and bridges", "playing hand"],
    "S033": ["ensemble", "instruments", "costume and setting"],
    "S034": ["building silhouette", "columns and roof", "surroundings"],
    "S035": ["reader and text", "altar", "reader clothing"],
    "S036": ["gift trays", "tray contents", "couple"],
    "S037": ["seating location", "seats", "drinks"],
    "S038": ["field surface", "material heaps", "field boundaries"],
    "S039": ["masonry", "tower silhouette", "doors"],
    "S040": ["hat silhouette", "hat material", "strap"],
    "S041": ["market", "flower merchandise", "goods and shoppers"],
    "S042": ["islands", "rock faces", "island tops and water"],
    "S043": ["printing action", "printing blocks", "paper and print"],
    "S044": ["branch and vase", "flowers", "surroundings"],
    "S045": ["altar", "ritual objects", "fruit offering"],
    "S046": ["walking surface", "supports", "handrail"],
    "S047": ["lower garment", "textile surface", "color bands"],
    "S048": ["wheel", "hands and clay", "vessel"],
    "S049": ["outer garment", "collar and fastening", "headwear"],
    "S050": ["envelope", "hands and people", "surroundings"],
}


def _human_name(identifier: str) -> str:
    return " ".join(identifier.replace("_or_", " / ").replace("_", " ").split())


def _requirement(pid: str, item: dict, index: int) -> dict:
    key = (pid, item["id"])
    if key in OPTIONAL:
        visibility, scoring, importance = "optional", "excluded", 1
    elif key in CONDITIONAL:
        visibility, scoring, importance = "check_if_visible", "conditional", 2
    else:
        visibility, scoring = "must_be_visible", "required"
        importance = 2 if key in SECONDARY or key in PROMPT_CONTEXT else 3

    if key in PROMPT_ACTION:
        requirement_type = "prompt_action"
    elif key in PROMPT_CONTEXT:
        requirement_type = "prompt_context"
    else:
        requirement_type = "identity"

    focus = str(item.get("part") or "").strip()
    if not focus:
        focus = FOCUS.get(pid, [])[index] if index < len(FOCUS.get(pid, [])) else "main subject"

    return {
        **item,
        "part": focus,
        "visual_evidence": item["description"],
        "visibility": visibility,
        "scoring": scoring,
        "requirement_type": requirement_type,
        "importance": importance,
        "repairable": scoring != "excluded",
    }


def build(source: dict, prompts: list[dict]) -> dict:
    prompt_map = {p["id"]: p for p in prompts}
    out = {}
    for pid in (f"S{i:03d}" for i in range(1, 51)):
        old = source[pid]
        required = [_requirement(pid, x, i) for i, x in enumerate(old.get("required", []))]
        core_ids = [x["id"] for x in required if x["scoring"] == "required" and x["importance"] == 3]
        if not core_ids:
            core_ids = [x["id"] for x in required if x["scoring"] == "required"]

        confusables = []
        for x in old.get("confusables", []):
            confusables.append({
                **x,
                "name_en": _human_name(x["id"]),
                "avoid_cues": [x["description"]],
                "distinguishing_required_ids": core_ids[:3],
            })

        positive = [x["visual_evidence"] for x in required
                    if x["scoring"] == "required" and x["importance"] == 3][:4]
        if not positive:
            positive = [x["visual_evidence"] for x in required if x["scoring"] == "required"][:4]
        optional = [x["visual_evidence"] for x in required if x["scoring"] != "required"]
        definition = f"A visual instance of {old['entity']}"
        if positive:
            definition += ", identified primarily by " + "; ".join(positive[:2])

        out[pid] = {
            "contract_version": "2.0",
            "prompt_id": pid,
            "prompt_text_en": prompt_map[pid]["text_en"],
            "entity": old["entity"],
            "entity_vi": old["entity_vi"],
            "main_entity": {
                "name_en": old["entity"],
                "name_vi": old["entity_vi"],
                "identity_definition": definition,
            },
            # Keep these two legacy keys so VietRepair can consume v2 unchanged.
            "required": required,
            "confusables": confusables,
            "generation_guidance": {
                "positive_cues": positive,
                "negative_cues": [x["name_en"] for x in confusables],
                "do_not_overconstrain": optional,
            },
            "verification": old.get("verification", {}),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default=str(ROOT.parent.parent / "ctig_contracts_verified.json"),
        help="verified S001-S050 JSON",
    )
    parser.add_argument("--prompts", default=str(ROOT / "data" / "prompts_simple.json"))
    parser.add_argument("--output", default=str(ROOT / "data" / "contracts_v2.json"))
    args = parser.parse_args()

    source = json.loads(Path(args.source).read_text(encoding="utf-8"))
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    result = build(source, prompts)
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(result)} contracts to {args.output}")


if __name__ == "__main__":
    main()
