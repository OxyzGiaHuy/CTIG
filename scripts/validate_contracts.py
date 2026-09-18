#!/usr/bin/env python3
"""Validate CTIG Visual Contract v2 without requiring third-party packages."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WIKI = re.compile(r"^https://(?:en|vi)\.wikipedia\.org/wiki/")
VISIBILITY = {"must_be_visible", "check_if_visible", "optional"}
SCORING = {"required", "conditional", "excluded"}
REQ_TYPES = {"identity", "prompt_action", "prompt_context"}


def validate(path: Path, prompt_path: Path) -> tuple[list[str], list[str], dict]:
    errors: list[str] = []
    warnings: list[str] = []
    data = json.loads(path.read_text(encoding="utf-8"))
    prompts = {x["id"]: x for x in json.loads(prompt_path.read_text(encoding="utf-8"))}
    expected = {f"S{i:03d}" for i in range(1, 51)}
    if set(data) != expected:
        errors.append(f"IDs must be exactly S001-S050; missing={sorted(expected-set(data))}, extra={sorted(set(data)-expected)}")

    counts = {"contracts": len(data), "required": 0, "confusables": 0,
              "required_scored": 0, "conditional": 0, "excluded": 0}
    for pid, contract in data.items():
        here = lambda msg: errors.append(f"{pid}: {msg}")  # noqa: E731
        if contract.get("prompt_id") != pid:
            here("prompt_id does not match its top-level key")
        if pid in prompts and contract.get("prompt_text_en") != prompts[pid].get("text_en"):
            here("prompt_text_en differs from data/prompts_simple.json")
        if contract.get("contract_version") != "2.0":
            here("contract_version must be 2.0")
        main = contract.get("main_entity") or {}
        if main.get("name_en") != contract.get("entity") or main.get("name_vi") != contract.get("entity_vi"):
            here("main_entity names disagree with legacy entity fields")

        required = contract.get("required") or []
        confusables = contract.get("confusables") or []
        counts["required"] += len(required)
        counts["confusables"] += len(confusables)
        req_ids = [x.get("id") for x in required]
        conf_ids = [x.get("id") for x in confusables]
        if len(req_ids) != len(set(req_ids)):
            here("duplicate required IDs")
        if len(conf_ids) != len(set(conf_ids)):
            here("duplicate confusable IDs")
        if set(req_ids) & set(conf_ids):
            here("an ID appears in both required and confusables")
        if not required or not confusables:
            here("required and confusables must both be non-empty")

        for item in required:
            iid = item.get("id", "?")
            for field in ("description", "visual_evidence", "source"):
                if not str(item.get(field, "")).strip():
                    here(f"required {iid}: empty {field}")
            if not WIKI.match(str(item.get("source", ""))):
                here(f"required {iid}: source is not an en/vi Wikipedia article")
            if item.get("visibility") not in VISIBILITY:
                here(f"required {iid}: invalid visibility")
            if item.get("scoring") not in SCORING:
                here(f"required {iid}: invalid scoring")
            if item.get("requirement_type") not in REQ_TYPES:
                here(f"required {iid}: invalid requirement_type")
            if item.get("importance") not in (1, 2, 3):
                here(f"required {iid}: importance must be 1, 2, or 3")
            expected_score = {"must_be_visible": "required", "check_if_visible": "conditional",
                              "optional": "excluded"}.get(item.get("visibility"))
            if item.get("scoring") != expected_score:
                here(f"required {iid}: visibility/scoring mismatch")
            counts[{"required": "required_scored", "conditional": "conditional",
                    "excluded": "excluded"}.get(item.get("scoring"), "excluded")] += 1
            if not str(item.get("part", "")).strip() and item.get("scoring") == "required":
                warnings.append(f"{pid}/{iid}: no observer part label")

        for item in confusables:
            iid = item.get("id", "?")
            if not WIKI.match(str(item.get("source", ""))):
                here(f"confusable {iid}: source is not an en/vi Wikipedia article")
            linked = item.get("distinguishing_required_ids") or []
            if not linked or not set(linked) <= set(req_ids):
                here(f"confusable {iid}: invalid distinguishing_required_ids")
            if not (item.get("avoid_cues") or []):
                here(f"confusable {iid}: avoid_cues is empty")

        guidance = contract.get("generation_guidance") or {}
        if not guidance.get("positive_cues"):
            here("generation_guidance.positive_cues is empty")

    return errors, warnings, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("contracts", nargs="?", default=str(ROOT / "data" / "contracts_v2.json"))
    parser.add_argument("--prompts", default=str(ROOT / "data" / "prompts_simple.json"))
    args = parser.parse_args()
    try:
        errors, warnings, counts = validate(Path(args.contracts), Path(args.prompts))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    for msg in warnings:
        print(f"WARNING: {msg}")
    for msg in errors:
        print(f"ERROR: {msg}", file=sys.stderr)
    print(json.dumps(counts, ensure_ascii=False))
    print(f"validation: {'FAIL' if errors else 'PASS'} ({len(errors)} errors, {len(warnings)} warnings)")
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
