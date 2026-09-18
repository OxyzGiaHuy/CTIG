"""Flow K / O / R — ba nhánh A · B(=I0) · C(=I1), một vòng sửa, cổng không-thoái-lui.

    python scripts/run_kor.py --config configs/vast_arms.yaml --ids S001,S002,S003 --run-name kor

    A  : prompt gốc P_orig                       -> SDXL -> ảnh
    B  : Culture-TRIP (P_orig ở đầu) = P_ct       -> SDXL -> I0
    C  : I0 -> K/O/R -> P1 -> SDXL img2img(I0, strength thấp) -> I1 ; cổng chọn I0 hay I1
    C+ref: như C nhưng sinh lại text2img CÙNG SEED + IP-Adapter ảnh thật (cột tham khảo)

Ba agent logic, CÙNG một backbone Mistral-Small-3.1-24B, khác system prompt:
    K  Prompt & Cultural Analyst  (text)  : Prompt Preservation Card + Cultural Evidence Card
    O  Blind Visual Observer      (VLM)   : CHỈ nhận ảnh, không prompt, không Wikipedia
    R  Gap Analyzer & Refiner     (text)  : ba loại lỗi, <=3 repair action DƯƠNG TÍNH
    G  = SDXL, không phải agent.

Không negative prompt ở MỌI nhánh (flow viết cho FLUX; giữ luật đó cả trên SDXL để ba nhánh chỉ khác
nhau ở prompt và ảnh khởi tạo). Cùng seed cho A/B/C. Mọi lời gọi agent được ghi nguyên văn vào
`kor.json` và `kor_transcript.md` — đó là "luồng giao tiếp" đưa vào supplementary.

Kiểm bằng MÁY, không tin lời dặn trong system prompt:
  - K: quote_vi của mỗi identity cue phải xuất hiện NGUYÊN VĂN trong bài Wikipedia đã chọn; không thì
       cue rơi xuống insufficient_evidence. Trần 3 / 2 / 2.
  - R: repair action chứa từ phủ định hay cụm khung hình -> loại. Tối đa 3.
  - Cổng: selection tính bằng luật từ fixed/regressions, không lấy chữ "selection" của model.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ctig.config import Config, set_dotted  # noqa: E402
from ctig.evaluation import ref_split  # noqa: E402
from ctig.pipeline import load_prompts  # noqa: E402
from ctig.session import Session  # noqa: E402
from scripts.run_loop_v2 import external_prompt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PHU_DINH = {"no", "not", "without", "never", "avoid", "don't", "dont", "instead", "remove", "none"}
KHUNG_HINH = ("close-up", "close up", "wide shot", "full body", "full-body", "top-down", "top down",
              "camera angle", "zoom", "crop", "lighting", "background", "composition", "framing", "portrait shot")

# ------------------------------------------------------------------ ghi lại mọi lời gọi
class SoTay:
    """Ghi system / user / response của từng lời gọi agent, đúng thứ tự — chính là luồng giao tiếp."""

    def __init__(self):
        self.dong: list[dict] = []

    def ghi(self, prompt_id, agent, buoc, system, user, response, images=None, giay=0.0):
        self.dong.append({"t": time.strftime("%H:%M:%S"), "prompt_id": prompt_id, "agent": agent, "buoc": buoc,
                          "system": system, "user": user, "images": images or [], "response": response,
                          "giay": round(giay, 1)})


def goi_text(agent, so_tay, pid, ten, buoc, system, user, schema, max_new_tokens=900):
    t0 = time.time()
    try:
        d = agent._complete(system, user, schema, max_new_tokens=max_new_tokens) or {}
    except Exception as exc:  # noqa: BLE001
        d = {"_loi": f"{type(exc).__name__}: {exc}"}
    so_tay.ghi(pid, ten, buoc, system, user, d, giay=time.time() - t0)
    return d


def goi_vlm(agent, so_tay, pid, ten, buoc, system, user, schema, image, max_new_tokens=900):
    t0 = time.time()
    try:
        d = agent.llm.complete_json(system, user, schema, images=[image], max_new_tokens=max_new_tokens) or {}
    except Exception as exc:  # noqa: BLE001
        d = {"_loi": f"{type(exc).__name__}: {exc}"}
    so_tay.ghi(pid, ten, buoc, system, user, d, images=[image], giay=time.time() - t0)
    return d


def _arr(): return {"type": "array", "items": {"type": "string"}}
def _norm(s): return re.sub(r"\s+", " ", str(s or "")).strip().lower()

# ------------------------------------------------------------------ K
K_SYSTEM = (
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
    "Answer in JSON only."
)
K_SCHEMA = {"type": "object", "properties": {
    "prompt_preservation": {"type": "object", "properties": {
        "main_entity": {"type": "object", "properties": {"vi": {"type": "string"}, "en": {"type": "string"}}},
        "subject_attributes": _arr(), "supporting_objects": _arr(), "background_and_scene": _arr(),
        "actions_and_relations": _arr(), "colors": _arr(), "visual_effects": _arr(),
        "time_weather_lighting": _arr(), "counts": {"type": "object"}, "text_in_image": _arr(),
        "must_preserve_verbatim": _arr()}},
    "cultural_evidence": {"type": "object", "properties": {
        "identity_cues": {"type": "array", "items": {"type": "object", "properties": {
            "cue_vi": {"type": "string"}, "cue_en": {"type": "string"}, "visual_check_vi": {"type": "string"},
            "quote_vi": {"type": "string"}, "source_url": {"type": "string"}}}},
        "conditional_cues": {"type": "array", "items": {"type": "object", "properties": {
            "cue_vi": {"type": "string"}, "cue_en": {"type": "string"}, "visual_check_vi": {"type": "string"},
            "quote_vi": {"type": "string"}, "source_url": {"type": "string"}}}},
        "positive_disambiguators": {"type": "array", "items": {"type": "object", "properties": {
            "target_appearance_vi": {"type": "string"}, "target_appearance_en": {"type": "string"},
            "source_url": {"type": "string"}}}},
        "discarded_nonvisual_facts": _arr(), "insufficient_evidence": _arr()}}},
    "required": ["prompt_preservation", "cultural_evidence"]}


def agent_K(agent, so_tay, pid, prompt_vi, prompt_en, wiki_pages, log):
    khoi = "\n\n".join(f"=== PASSAGE [{p['lang']}] {p['title']} — {p['url']} ===\n{p['text'][:12000]}"
                       for p in wiki_pages)
    user = (f"PROMPT VI: {prompt_vi}\nPROMPT EN: {prompt_en}\n\nWIKIPEDIA PASSAGES (the only allowed source "
            f"for cultural cues):\n{khoi}\n\nReturn the two cards as JSON.")
    d = goi_text(agent, so_tay, pid, "K", "cards", K_SYSTEM, user, K_SCHEMA, max_new_tokens=1400)
    ce = d.get("cultural_evidence") or {}
    # ---- kiểm nguyên văn bằng máy
    vi_text = _norm(" ".join(p["text"] for p in wiki_pages if p["lang"] == "vi"))
    urls = {p["url"] for p in wiki_pages}
    loai = []
    for k in ("identity_cues", "conditional_cues"):
        giu = []
        for c in (ce.get(k) or []):
            q = _norm(c.get("quote_vi"))
            if len(q) >= 12 and q in vi_text and (c.get("source_url") in urls):
                giu.append(c)
            else:
                loai.append(f"{k}: '{str(c.get('cue_vi'))[:50]}' — quote không nguyên văn trong bài / URL lạ")
        ce[k] = giu[:3 if k == "identity_cues" else 2]
    ce["positive_disambiguators"] = (ce.get("positive_disambiguators") or [])[:2]
    ce["insufficient_evidence"] = (ce.get("insufficient_evidence") or []) + loai
    d["cultural_evidence"] = ce
    so_tay.ghi(pid, "K", "may_kiem", "(kiểm bằng máy: quote_vi phải nguyên văn trong bài vi)", "",
               {"giu_identity": len(ce["identity_cues"]), "giu_conditional": len(ce["conditional_cues"]), "loai": loai})
    log(f"  [K] identity {len(ce['identity_cues'])} · conditional {len(ce['conditional_cues'])} · "
        f"disambig {len(ce['positive_disambiguators'])} · loại vì không nguyên văn: {len(loai)}")
    return d

# ------------------------------------------------------------------ O
O_SYSTEM = (
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
O_SCHEMA = {"type": "object", "properties": {
    "main_subjects": _arr(), "clothing_and_appearance": _arr(), "foreground_objects": _arr(),
    "supporting_objects": _arr(), "background": _arr(), "actions": _arr(), "spatial_relations": _arr(),
    "colors": _arr(), "counts": {"type": "object"}, "visible_effects": _arr(), "uncertain": _arr()},
    "required": ["main_subjects"]}


def agent_O(agent, so_tay, pid, image, buoc, log):
    d = goi_vlm(agent, so_tay, pid, "O", buoc, O_SYSTEM, "Describe this photograph. Return JSON.", O_SCHEMA, image)
    n = sum(len(v) for v in d.values() if isinstance(v, list))
    log(f"  [O·{buoc}] {n} quan sát · chủ thể: {'; '.join((d.get('main_subjects') or [])[:2])[:80]}")
    return d

# ------------------------------------------------------------------ R
R_SYSTEM = (
    "You are the Gap Analyzer & Prompt Refiner. You compare a blind visual report of a generated image "
    "against (a) the Prompt Preservation Card and (b) the Cultural Evidence Card.\n"
    "Classify: missing_prompt_explicit (asked by the prompt, not in the report); missing_cultural_identity "
    "(identity cue not evidenced in the report); contradictions (report shows something incompatible); "
    "already_satisfied; uncertain_no_repair (the report is unsure — do NOT repair these).\n"
    "Priority for repair: 1 details stated in the prompt; 2 supporting objects, background, relations; "
    "3 identity cues; 4 conditional cues. At most 3 repair_actions.\n"
    "Each repair_action is ONE positive English instruction (max 18 words) describing exactly what should be "
    "visibly present. Never use negation (no/not/without/avoid). Never name the wrong object. Never change "
    "framing, camera angle, lighting or composition. Do not restate things already satisfied. JSON only."
)
R_SCHEMA = {"type": "object", "properties": {
    "missing_prompt_explicit": _arr(), "missing_cultural_identity": _arr(), "contradictions": _arr(),
    "already_satisfied": _arr(), "uncertain_no_repair": _arr(), "repair_actions": _arr()},
    "required": ["repair_actions"]}


def _sach(a: str) -> str | None:
    low = " " + _norm(a) + " "
    if set(low.split()) & PHU_DINH or any(k in low for k in KHUNG_HINH):
        return None
    return " ".join(str(a).split())


def agent_R(agent, so_tay, pid, prompt_en, cards, report, log):
    user = (f"ORIGINAL PROMPT: {prompt_en}\n\nPROMPT PRESERVATION CARD:\n"
            f"{json.dumps(cards.get('prompt_preservation'), ensure_ascii=False)}\n\nCULTURAL EVIDENCE CARD:\n"
            f"{json.dumps(cards.get('cultural_evidence'), ensure_ascii=False)}\n\nBLIND VISUAL REPORT OF THE IMAGE:\n"
            f"{json.dumps(report, ensure_ascii=False)}\n\nReturn the gap analysis as JSON.")
    d = goi_text(agent, so_tay, pid, "R", "gap", R_SYSTEM, user, R_SCHEMA, max_new_tokens=900)
    tho = [str(x) for x in (d.get("repair_actions") or [])]
    sach = [y for y in (_sach(x) for x in tho) if y][:3]
    bo = [x for x in tho if _sach(x) is None]
    d["repair_actions"] = sach
    so_tay.ghi(pid, "R", "may_kiem", "(kiểm bằng máy: bỏ action có phủ định / khung hình; trần 3)", "",
               {"giu": sach, "bo": bo})
    log(f"  [R] thiếu-prompt {len(d.get('missing_prompt_explicit') or [])} · thiếu-văn-hoá "
        f"{len(d.get('missing_cultural_identity') or [])} · action giữ {len(sach)}" + (f" · bỏ {len(bo)}" if bo else ""))
    return d


GATE_SYSTEM = (
    "You compare two blind visual reports of two images (BEFORE and AFTER a repair) against the same two "
    "cards and the list of repair actions that were attempted.\n"
    "fixed: attempted repairs now evidenced in AFTER but not in BEFORE. regressions: things from the "
    "Prompt Preservation Card or already-satisfied list that are present in BEFORE but missing or "
    "contradicted in AFTER; mark each with severity 'severe' (main entity, counts, a required object lost) "
    "or 'minor'. Be conservative: if unsure, it is not fixed. JSON only."
)
GATE_SCHEMA = {"type": "object", "properties": {
    "fixed": _arr(), "regressions": {"type": "array", "items": {"type": "object", "properties": {
        "what": {"type": "string"}, "severity": {"type": "string"}}}},
    "unchanged_issues": _arr()}, "required": ["fixed", "regressions"]}


def cong_gate(agent, so_tay, pid, cards, rep0, rep1, actions, log, nhan="C"):
    user = (f"CARDS:\n{json.dumps({k: cards.get(k) for k in ('prompt_preservation', 'cultural_evidence')}, ensure_ascii=False)}\n\n"
            f"REPAIR ACTIONS ATTEMPTED: {json.dumps(actions, ensure_ascii=False)}\n\n"
            f"BEFORE (I0) REPORT:\n{json.dumps(rep0, ensure_ascii=False)}\n\nAFTER (I1) REPORT:\n"
            f"{json.dumps(rep1, ensure_ascii=False)}\n\nReturn JSON.")
    d = goi_text(agent, so_tay, pid, "R", f"gate_{nhan}", GATE_SYSTEM, user, GATE_SCHEMA, max_new_tokens=700)
    fixed = [str(x) for x in (d.get("fixed") or [])]
    regs = [r for r in (d.get("regressions") or []) if isinstance(r, dict)]
    nang = [r for r in regs if _norm(r.get("severity")).startswith("sev")]
    # LUẬT, không phải lời model: có regression nặng -> I0; không sửa được gì -> I0; còn lại -> I1
    if nang:
        chon, ly_do = "I0", f"{len(nang)} regression nghiêm trọng"
    elif not fixed:
        chon, ly_do = "I0", "không sửa được lỗi nào"
    else:
        chon, ly_do = "I1", f"sửa được {len(fixed)}, không regression nghiêm trọng"
    out = {"fixed": fixed, "regressions": regs, "selection": chon, "ly_do": ly_do}
    so_tay.ghi(pid, "GATE", f"luat_{nhan}", "(luật cổng, tính bằng máy)", "", out)
    log(f"  [cổng·{nhan}] fixed {len(fixed)} · regression {len(regs)} (nặng {len(nang)}) -> chọn {chon} ({ly_do})")
    return out


def dung_P1(p_ct: str, actions: list[str]) -> str:
    """P1 = P_ct nguyên văn (P_orig đã nằm ở đầu) + preserve clause + <=3 action dương tính. Không negative."""
    if not actions:
        return p_ct
    ds = "\n".join(f"{i + 1}. {a.rstrip('.')}." for i, a in enumerate(actions))
    return (f"{p_ct.strip()}\n\nPreserve the current subject, composition, setting, colors, and all correctly "
            f"rendered details. Make these additions clearly visible:\n{ds}")

# ------------------------------------------------------------------ lưới + transcript
def _font(size, bold=False):
    from PIL import ImageFont
    n = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    p = f"/usr/share/fonts/truetype/dejavu/{n}"
    return ImageFont.truetype(p, size) if Path(p).exists() else ImageFont.load_default()


def ve_luoi(hang, cot, out_png, cell=300):
    from PIL import Image, ImageDraw
    gut, head, pad, cap = 70, 30, 6, 22
    cw, ch = cell + pad, cell + cap + pad
    W, H = gut + len(cot) * cw + pad, head + len(hang) * ch + pad
    im = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(im)
    for i, c in enumerate(cot):
        d.text((gut + i * cw + 3, 8), c, font=_font(14, True), fill=(20, 20, 20))
    for r, (pid, o, ghi) in enumerate(hang):
        y = head + r * ch
        d.text((4, y + 6), pid, font=_font(13), fill=(20, 20, 20))
        for i, c in enumerate(cot):
            p = o.get(c); x = gut + i * cw
            if not p or not Path(p).exists():
                d.text((x + 6, y + 6), "—", font=_font(13), fill=(150, 150, 150)); continue
            t = Image.open(p).convert("RGB"); t.thumbnail((cell, cell))
            im.paste(t, (x + (cell - t.width) // 2, y))
            if ghi.get(c):
                d.text((x + 3, y + cell + 3), ghi[c][:44], font=_font(11), fill=(110, 110, 110))
        d.line([(0, y - 2), (W, y - 2)], fill=(225, 225, 225))
    im.save(out_png)


def transcript_md(so_tay: SoTay, out_md: Path):
    L = ["# Luồng giao tiếp K / O / R\n", "Mỗi lời gọi ghi nguyên văn: system prompt (in một lần cho mỗi agent), "
         "user prompt, và JSON trả về. Dòng `may_kiem` / `luat_*` là bước kiểm bằng máy, không phải model.\n"]
    da_in = set()
    for pid in dict.fromkeys(x["prompt_id"] for x in so_tay.dong):
        L.append(f"\n---\n\n## {pid}\n")
        for x in [y for y in so_tay.dong if y["prompt_id"] == pid]:
            L.append(f"\n### {x['agent']} · {x['buoc']}  ·  {x['t']}" + (f"  ·  {x['giay']}s" if x["giay"] else "") + "\n")
            if x["system"] and (x["agent"], x["system"][:40]) not in da_in and not x["system"].startswith("("):
                da_in.add((x["agent"], x["system"][:40]))
                L.append(f"\n<details><summary>system prompt của {x['agent']}</summary>\n\n```\n{x['system']}\n```\n</details>\n")
            elif x["system"].startswith("("):
                L.append(f"\n*{x['system']}*\n")
            if x["images"]:
                L.append(f"\nảnh vào: `{Path(x['images'][0]).name}`\n")
            if x["user"]:
                u = x["user"] if len(x["user"]) < 3500 else x["user"][:3500] + f"\n… [cắt, tổng {len(x['user'])} ký tự]"
                L.append(f"\n<details><summary>user prompt</summary>\n\n```\n{u}\n```\n</details>\n")
            L.append(f"\n```json\n{json.dumps(x['response'], ensure_ascii=False, indent=1)}\n```\n")
    out_md.write_text("".join(L), encoding="utf-8")

# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ids", default="S001,S002,S003")
    ap.add_argument("--model", default="sdxl_base")
    ap.add_argument("--run-name", default="kor")
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--strength", type=float, default=0.35)
    ap.add_argument("--wiki", default=str(ROOT / "data" / "wiki_curated" / "S001_S003.json"))
    ap.add_argument("--no-cref", action="store_true", help="bỏ cột C+ref")
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args(argv)

    ov: dict = {}
    for kv in a.set:
        k, _, v = kv.partition("="); set_dotted(ov, k, v)
    set_dotted(ov, "t2i.render", "bare"); set_dotted(ov, "multigen.adaptive.enabled", "false")
    set_dotted(ov, "multigen.n_candidates", "1"); set_dotted(ov, "multigen.keep_loaded", "1")
    cfg = Config.load(a.config, ov)

    wiki = json.loads(Path(a.wiki).read_text(encoding="utf-8"))
    allp = {p.id: p for p in load_prompts(cfg.prompts_path)}
    ids = [i.strip() for i in a.ids.split(",") if i.strip() in allp]
    run_dir = Path(cfg.runs_dir) / a.run_name; run_dir.mkdir(parents=True, exist_ok=True)
    log = lambda *x: print(*x, flush=True)  # noqa: E731
    so_tay = SoTay()

    from ctig.models import loader as ML, registry as REG
    from ctig.stages import multigen as mg
    from ctig.stages.generation import DiffusersGenerator
    REGISTRY = getattr(REG, "MODELS", None) or getattr(REG, "REGISTRY")

    cot = ["A", "B (I0)", "C (I1)"] + ([] if a.no_cref else ["C+ref"])
    hang, tong = [], []
    shared = [None]
    for pid in ids:
        pr = allp[pid]
        out_dir = run_dir / pid; out_dir.mkdir(parents=True, exist_ok=True)
        log(f"\n================ {pid} · {pr.text_vi[:60]} ================")
        s = Session(cfg, pr, run_dir=out_dir, log=lambda *x: None)
        if shared[0] is not None:
            s._agent = shared[0]
        s.skip_grounding(); s.spec()
        p_orig = pr.text_en.strip()
        p_ct, _ = external_prompt("culture_trip", pid)
        p_ct = " ".join((p_ct or p_orig).split())
        if not p_ct.startswith(p_orig):
            log(f"  [!] Culture-TRIP không bắt đầu bằng prompt gốc -> chèn P_orig lên đầu")
            p_ct = f"{p_orig} {p_ct}"
        gen0, _ = s.genspec()
        dem = {"A": 0, "B": 0, "C": 0, "C+ref": 0}

        def sinh(prompt, sub, nhanh, refs=None, seed=None):
            dem[nhanh] += 1
            rf = refs or []
            g = replace(gen0, prompt_terms=[prompt], negative_terms=[], seed=a.seed if seed is None else seed,
                        iteration=0, ip_adapter_image=(rf or None), ip_adapter_scale=cfg.multigen.ref_scale)
            r = mg.run(g, s.spec()[0], s.kb, [a.model + "+ref" if rf else a.model], cfg.multigen, out_dir / sub,
                       clip=s.clip, itm=None, t2i_cfg=cfg.t2i, prompt_en=prompt, log=lambda *x: None,
                       ref_images=rf, force_refs=bool(rf))
            return next((rr.output.candidates[0].path for rr in r.runs if rr.output and rr.output.candidates), None)

        def sinh_i2i(prompt, init, sub):
            """I1 = img2img(I0, P1, strength thấp): giữ bố cục, chỉ sửa chi tiết. Không IP-Adapter, không negative."""
            dem["C"] += 1
            mspec = REGISTRY[a.model]
            pipe = ML.load_pipeline(mspec, cfg.multigen.device, cfg.multigen.cpu_offload, log=lambda *x: None,
                                    scheduler=cfg.multigen.scheduler, keep_loaded=1)
            g = DiffusersGenerator(pipe, a.model, negative_ok=True, family=mspec.family, long_prompt=True,
                                   log=lambda *x: None, init_image=init, init_strength=a.strength)
            gs = replace(gen0, prompt_terms=[prompt], negative_terms=[], seed=a.seed, iteration=0,
                         n_candidates=1, width=mspec.width, height=mspec.height, steps=mspec.steps,
                         guidance=mspec.guidance)
            out = g.generate(gs, s.spec()[0], s.kb, out_dir / sub)
            return out.candidates[0].path if out and out.candidates else None

        # ---- A và B (= I0)
        anh_A = sinh(p_orig, "A", "A"); log(f"  [A] {anh_A}")
        i0 = sinh(p_ct, "B", "B"); log(f"  [B=I0] {i0}")
        shared[0] = s.agent
        if not i0:
            log("  không có I0 -> bỏ prompt"); continue

        # ---- K: hai card (text, không nhìn ảnh)
        cards = agent_K(s.agent, so_tay, pid, pr.text_vi, p_orig, wiki.get(pid, []), log)
        # ---- O: quan sát mù I0
        rep0 = agent_O(s.agent, so_tay, pid, i0, "I0", log)
        # ---- R: phân tích lỗ hổng -> action
        gap = agent_R(s.agent, so_tay, pid, p_orig, cards, rep0, log)
        actions = gap.get("repair_actions") or []
        p1 = dung_P1(p_ct, actions)
        so_tay.ghi(pid, "P1", "prompt", "(P1 = P_ct nguyên văn + preserve clause + action; không negative)", "", {"P1": p1})

        anh = {"A": anh_A, "B (I0)": i0}
        ghi = {"B (I0)": "= I0", "A": "prompt gốc"}
        gate_C = gate_Cref = None
        if not actions:
            log("  [C] no-op: R không có action hợp lệ -> I1 = I0")
            anh["C (I1)"] = i0; ghi["C (I1)"] = "no-op (= I0)"
            if not a.no_cref:
                anh["C+ref"] = i0; ghi["C+ref"] = "no-op (= I0)"
        else:
            # ---- C: img2img từ I0
            i1 = sinh_i2i(p1, i0, "C_i2i")
            log(f"  [C] img2img strength {a.strength} -> {i1}")
            rep1 = agent_O(s.agent, so_tay, pid, i1, "I1", log) if i1 else {}
            gate_C = cong_gate(s.agent, so_tay, pid, cards, rep0, rep1, actions, log, "C") if i1 else None
            anh["C (I1)"] = i1 or i0
            ghi["C (I1)"] = f"chọn {gate_C['selection']}" if gate_C else "sinh hỏng"
            # ---- C+ref: cùng seed text2img + IP-Adapter ảnh thật (cột tham khảo, không phải flow chính)
            if not a.no_cref:
                refs, _ = ref_split(cfg.retrieval.ref_dir, pid, 5)
                refs = refs[:cfg.multigen.ref_images]
                if refs and cfg.multigen.ref_crop:
                    refs = s.crop_refs(refs, s.spec()[0])
                i1r = sinh(p1, "C_ref", "C+ref", refs=refs) if refs else None
                log(f"  [C+ref] cùng seed, IP-Adapter {len(refs)} ảnh thật -> {i1r}")
                rep1r = agent_O(s.agent, so_tay, pid, i1r, "I1ref", log) if i1r else {}
                gate_Cref = cong_gate(s.agent, so_tay, pid, cards, rep0, rep1r, actions, log, "Cref") if i1r else None
                anh["C+ref"] = i1r or i0
                ghi["C+ref"] = f"chọn {gate_Cref['selection']}" if gate_Cref else "không ref"

        rec = {"prompt_id": pid, "prompt_vi": pr.text_vi, "P_orig": p_orig, "P_ct": p_ct, "P1": p1, "seed": a.seed,
               "model": a.model, "strength_i2i": a.strength, "images": anh, "so_lan_sinh": dem,
               "cards": cards, "report_I0": rep0, "gap": gap, "actions": actions,
               "gate_C": gate_C, "gate_Cref": gate_Cref}
        (out_dir / "kor.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        tong.append(rec); hang.append((pid, anh, ghi))
        ve_luoi(hang, cot, run_dir / "kor_grid.png")
        (run_dir / "kor.json").write_text(json.dumps({"don_vi": tong, "giao_tiep": so_tay.dong},
                                                     ensure_ascii=False, indent=1), encoding="utf-8")
        transcript_md(so_tay, run_dir / "kor_transcript.md")
        log(f"  sinh: {dem} · -> {out_dir / 'kor.json'}")

    log("\nKOR_DONE")


if __name__ == "__main__":
    main()
