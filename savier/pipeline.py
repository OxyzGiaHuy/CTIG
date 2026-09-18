"""SAVIER end-to-end for one prompt set: arms A · B(=I0) · C(=I1), transcript, contact sheet.

    A   P0 (original English prompt)                 -> G            seed s
    B   P_ct = P0 + Culture-TRIP expansion (verbatim)  -> G  = I0     seed s
    C   I0 -> C(urator) -> O(bserver) -> R(efiner) -> P1 -> G + reference conditioning = I1, same seed s

I1 is always the method's output. There is no acceptance gate: evaluation is done afterwards by an
independent evaluator (scripts/eval_metrics.py), never by the agents themselves.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .agents import Transcript, build_P1, curator, observer, refiner


def load_prompts(path): return {p["id"]: p for p in json.loads(Path(path).read_text(encoding="utf-8"))}


def culture_trip_prompt(dir_: str | Path, pid: str) -> str:
    f = Path(dir_) / f"{pid}.json"
    return " ".join(str(json.loads(f.read_text(encoding="utf-8")).get("refined_prompt", "")).split()) if f.exists() else ""


def reference_images(refs_dir: str | Path | None, pid: str, k: int = 2) -> list[str]:
    """Curated real photographs used ONLY for conditioning (`selected/<pid>/`). Held-out evaluation photos
    live in `candidates/<pid>/` and are never passed here. Photos are third-party and not redistributed."""
    if not refs_dir: return []
    d = Path(refs_dir) / "selected" / pid
    return sorted(str(p) for p in d.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))[:k] if d.exists() else []


def run(llm, gen, prompts: dict, ids: list[str], wiki: dict, ct_dir, out_dir, refs_dir=None, seed=5000,
        contracts: dict | None = None, text_only_arm=False, log=print) -> dict:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tr, units = Transcript(), []
    for pid in ids:
        pr = prompts[pid]; d = out_dir / pid; d.mkdir(exist_ok=True)
        p0 = pr["text_en"].strip(); p_ct = culture_trip_prompt(ct_dir, pid) or p0
        if not p_ct.startswith(p0): p_ct = f"{p0} {p_ct}"
        log(f"\n===== {pid} · {pr['text_vi']} =====")
        a = gen.generate(p0, seed, d / "A.png")
        i0 = gen.generate(p_ct, seed, d / "I0.png")
        kw = list(pr.get("gold_entities") or pr.get("entities") or []) + [w for w in pr["text_vi"].split() if len(w) > 3]
        if contracts and pid in contracts:
            c = contracts[pid]; kw += [c.get("entity_vi", "")] + [str(r.get("part", "")).replace("|", " ") for r in c.get("required", [])]
        cards = curator(llm, tr, pid, pr["text_vi"], p0, wiki.get(pid, []), kw, log)
        rep0 = observer(llm, tr, pid, i0, "I0", log)
        expansion = p_ct[len(p0):].strip()
        gap = refiner(llm, tr, pid, p0, expansion, cards, rep0, i0, log)
        actions = gap["repair_actions"]
        p1 = build_P1(p_ct, p0, actions, cards["prompt_preservation"], gap["drop_phrases"])
        tr.log(pid, "P1", "prompt", "(P1 = P0 verbatim + trimmed expansion + Keep clause + <=3 actions; no negative prompt)", "", {"P1": p1})
        refs = reference_images(refs_dir, pid)
        i1 = gen.generate(p1, seed, d / "I1.png", refs=refs) if actions else i0
        images = {"A": a, "I0": i0, "I1": i1}
        if text_only_arm and actions:
            images["I1_text_only"] = gen.generate(p1, seed, d / "I1_text_only.png")
        u = {"prompt_id": pid, "prompt_vi": pr["text_vi"], "P0": p0, "P_ct": p_ct, "P1": p1, "seed": seed, "refs": refs,
             "cards": cards, "report_I0": rep0, "gap": gap, "actions": actions, "noop": not actions, "images": images}
        (d / "unit.json").write_text(json.dumps(u, ensure_ascii=False, indent=1), encoding="utf-8")
        units.append(u)
        (out_dir / "savier.json").write_text(json.dumps({"units": units, "transcript": tr.rows}, ensure_ascii=False, indent=1), encoding="utf-8")
        write_transcript(tr, out_dir / "transcript.md"); contact_sheet(units, out_dir / "grid.png")
    return {"units": units, "transcript": tr.rows}


def write_transcript(tr: Transcript, path: Path):
    L, seen = ["# SAVIER agent transcript\n"], set()
    for pid in dict.fromkeys(r["prompt_id"] for r in tr.rows):
        L.append(f"\n---\n\n## {pid}\n")
        for r in [x for x in tr.rows if x["prompt_id"] == pid]:
            L.append(f"\n### {r['agent']} · {r['step']} · {r['t']}" + (f" · {r['seconds']}s" if r["seconds"] else "") + "\n")
            if r["system"].startswith("("): L.append(f"\n*{r['system']}*\n")
            elif (r["agent"], r["system"][:40]) not in seen:
                seen.add((r["agent"], r["system"][:40])); L.append(f"\n<details><summary>system prompt ({r['agent']})</summary>\n\n```\n{r['system']}\n```\n</details>\n")
            if r["images"]: L.append(f"\nimage: `{Path(r['images'][0]).name}`\n")
            if r["user"]:
                u = r["user"] if len(r["user"]) < 3500 else r["user"][:3500] + f"\n… [truncated, {len(r['user'])} chars]"
                L.append(f"\n<details><summary>user prompt</summary>\n\n```\n{u}\n```\n</details>\n")
            L.append(f"\n```json\n{json.dumps(r['reply'], ensure_ascii=False, indent=1)}\n```\n")
    path.write_text("".join(L), encoding="utf-8")


def contact_sheet(units, out_png: Path, cell=300, model_label="model"):
    from PIL import Image, ImageDraw, ImageFont
    import textwrap
    cols = ["A", "I0", "I1"] + (["I1_text_only"] if any("I1_text_only" in u["images"] for u in units) else [])
    labels = {"A": model_label, "I0": f"{model_label} + refined prompt", "I1": f"{model_label} + SAVIER (ours)", "I1_text_only": f"{model_label} + SAVIER (ours, text only)"}
    try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError: font = ImageFont.load_default()
    gut, head, pad = 210, 30, 6; cw, ch = cell + pad, cell + pad
    im = Image.new("RGB", (gut + len(cols) * cw + pad, head + len(units) * ch + pad), "white"); dr = ImageDraw.Draw(im)
    for i, c in enumerate(cols): dr.text((gut + i * cw + 3, 8), labels[c], font=font, fill=(20, 20, 20))
    for r, u in enumerate(units):
        y = head + r * ch
        for j, line in enumerate(textwrap.wrap(f"{u['prompt_id']}: {u['prompt_vi']}", 30)[:9]): dr.text((4, y + 6 + j * 15), line, font=font, fill=(20, 20, 20))
        for i, c in enumerate(cols):
            p = u["images"].get(c)
            if p and Path(p).exists():
                t = Image.open(p).convert("RGB"); t.thumbnail((cell, cell)); im.paste(t, (gut + i * cw + (cell - t.width) // 2, y))
    im.save(out_png)
