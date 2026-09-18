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
from ctig.evaluation import ref_similarity, ref_split  # noqa: E402
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
    "- Cues must describe the MAIN OBJECT's own appearance, or HOW it sits on the body (resting on one "
    "shoulder, worn over trousers, tied at the waist) when the source says so. Utensils, containers, "
    "serving temperature and preparation steps are not cues.\n"
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


def cat_bai(text: str, tu_khoa: list[str], toi_da: int = 4000) -> str:
    """Giữ những câu chứa danh từ thực thể, theo thứ tự, tới `toi_da` ký tự; không câu nào khớp thì lấy đầu bài.

    Vì sao: bản đầu đưa nguyên bài 12k ký tự, K trả JSON dài quá `max_new_tokens` rồi CỤT ở giữa, parse
    hỏng 3 lần liền ở S001 và S003 (170 s mỗi prompt) -> Cultural Evidence Card rỗng ở cả 3 prompt.
    Bài 6k (S002) thì ra được. Cắt còn ~4k quanh danh từ thực thể là đủ chỗ cho quote nguyên văn.
    """
    cau = re.split(r"(?<=[.!?])\s+", text)
    tk = [k.lower() for k in tu_khoa if k]
    chon, n = [], 0
    for c in cau:
        if any(k in c.lower() for k in tk) and n + len(c) <= toi_da:
            chon.append(c); n += len(c) + 1
    if len(" ".join(chon)) < 800:
        return text[:toi_da]
    return " ".join(chon)


K_SYSTEM_A = ("You are the Prompt Analyst. From the prompt ONLY, produce prompt_preservation: everything the prompt "
              "asks for, grouped. Keep meaning intact; the English wording in must_preserve_verbatim must be copied "
              "exactly. Needs no citation. JSON only.")
K_SCHEMA_A = {"type": "object", "properties": {"prompt_preservation": K_SCHEMA["properties"]["prompt_preservation"]},
              "required": ["prompt_preservation"]}
K_SCHEMA_B = {"type": "object", "properties": {"cultural_evidence": K_SCHEMA["properties"]["cultural_evidence"]},
              "required": ["cultural_evidence"]}


def agent_K(agent, so_tay, pid, prompt_vi, prompt_en, wiki_pages, log, tu_khoa=None):
    # Lời gọi 1: Preservation Card, chỉ từ prompt (ngắn, không thể cụt)
    ua = f"PROMPT VI: {prompt_vi}\nPROMPT EN: {prompt_en}\n\nReturn prompt_preservation as JSON."
    da = goi_text(agent, so_tay, pid, "K", "card_preservation", K_SYSTEM_A, ua, K_SCHEMA_A, max_new_tokens=700)
    # Lời gọi 2: Cultural Evidence Card, từ bài Wikipedia đã CẮT quanh danh từ thực thể
    # Trần 4k cho TỔNG các bài (S005 có vi + en, mỗi bài 4k -> 8,4k vào, JSON ra lại cụt). Bài vi trước.
    tran = 4000; phan = []
    for pg in sorted(wiki_pages, key=lambda x: 0 if x["lang"] == "vi" else 1):
        if tran <= 600: break
        doan = cat_bai(pg["text"], tu_khoa or [], toi_da=tran)
        phan.append(f"=== PASSAGE [{pg['lang']}] {pg['title']} — {pg['url']} ===\n{doan}"); tran -= len(doan)
    khoi = "\n\n".join(phan)
    ub = (f"PROMPT VI: {prompt_vi}\nPROMPT EN: {prompt_en}\n\nWIKIPEDIA PASSAGES (the only allowed source "
          f"for cultural cues):\n{khoi}\n\nReturn cultural_evidence as JSON. quote_vi must be copied verbatim.")
    db = goi_text(agent, so_tay, pid, "K", "card_cultural", K_SYSTEM, ub, K_SCHEMA_B, max_new_tokens=2200)
    d = {"prompt_preservation": da.get("prompt_preservation") or {}, "cultural_evidence": db.get("cultural_evidence") or {},
         "_loi": [x.get("_loi") for x in (da, db) if x.get("_loi")]}
    ce = d["cultural_evidence"]
    # ---- kiểm nguyên văn bằng máy
    vi_text = _norm(" ".join(p["text"] for p in wiki_pages if p["lang"] == "vi"))
    urls = {p["url"] for p in wiki_pages}
    loai = []
    for k in ("identity_cues", "conditional_cues"):
        giu = []
        for c in (ce.get(k) or []):
            q = _norm(c.get("quote_vi"))
            # Quote nguyên văn CHƯA đủ: S003 K viết cue "stone bowl" kèm quote (nguyên văn) về nước dùng đun
            # sôi — quote có thật nhưng KHÔNG nói gì về bát đá; R rồi đòi "Include a stone bowl". Nên đòi
            # thêm: ít nhất một từ nội dung của cue_vi phải nằm trong quote_vi (kiểm suy diễn thô).
            tu_cue = {w for w in _norm(c.get("cue_vi")).split() if len(w) > 2}
            khop = bool(tu_cue & set(q.split()))
            if len(q) >= 12 and q in vi_text and (c.get("source_url") in urls) and khop:
                giu.append(c)
            elif not khop and q:
                loai.append(f"{k}: '{str(c.get('cue_vi'))[:50]}' — quote có thật nhưng không nói về cue")
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
    "visibly present, including WHERE on the body or scene it sits when that matters (e.g. 'a bamboo pole "
    "resting across one shoulder with a basket hanging from each end'). Never use negation (no/not/without/avoid). Never name the wrong object. Never change "
    "framing, camera angle, lighting or composition. Do not restate things already satisfied.\n"
    "drop_phrases: copy VERBATIM any phrase from the EXPANSION TEXT (never from the original prompt) that "
    "contradicts the Cultural Evidence Card or pushes toward a look-alike object — e.g. a cut or garment "
    "term that belongs to another culture. Empty list if none. JSON only."
)
R_SCHEMA = {"type": "object", "properties": {
    "missing_prompt_explicit": _arr(), "missing_cultural_identity": _arr(), "contradictions": _arr(),
    "already_satisfied": _arr(), "uncertain_no_repair": _arr(), "repair_actions": _arr(),
    "drop_phrases": _arr()},
    "required": ["repair_actions"]}


def _sach(a: str) -> str | None:
    low = " " + _norm(a) + " "
    if set(low.split()) & PHU_DINH or any(k in low for k in KHUNG_HINH):
        return None
    return " ".join(str(a).split())


def agent_R(agent, so_tay, pid, prompt_en, cards, report, log, mo_rong=""):
    user = (f"ORIGINAL PROMPT: {prompt_en}\n\nEXPANSION TEXT (added by Culture-TRIP, may be trimmed):\n{mo_rong}\n\n"
            f"PROMPT PRESERVATION CARD:\n"
            f"{json.dumps(cards.get('prompt_preservation'), ensure_ascii=False)}\n\nCULTURAL EVIDENCE CARD:\n"
            f"{json.dumps(cards.get('cultural_evidence'), ensure_ascii=False)}\n\nBLIND VISUAL REPORT OF THE IMAGE:\n"
            f"{json.dumps(report, ensure_ascii=False)}\n\nReturn the gap analysis as JSON.")
    d = goi_text(agent, so_tay, pid, "R", "gap", R_SYSTEM, user, R_SCHEMA, max_new_tokens=900)
    tho = [str(x) for x in (d.get("repair_actions") or [])]
    sach = [y for y in (_sach(x) for x in tho) if y][:3]
    bo = [x for x in tho if _sach(x) is None]
    d["repair_actions"] = sach
    # Cụm bị cắt phải NẰM NGUYÊN VĂN trong phần mở rộng (không bao giờ là P0), và không quá 3 cụm.
    # Vì sao: B (Culture-TRIP) ra qipao ở cả SDXL và FLUX trong khi A (prompt gốc) ra áo dài — phần mở rộng
    # đang kéo về vật sai; trên FLUX nặng hơn vì T5 đọc hết prompt dài. Cắt nó là đánh vào nguyên nhân.
    drop = []
    for x in (d.get("drop_phrases") or []):
        x = " ".join(str(x).split())
        if len(x) >= 6 and x.lower() in mo_rong.lower() and x.lower() not in prompt_en.lower():
            drop.append(x)
    d["drop_phrases"] = drop[:3]
    so_tay.ghi(pid, "R", "may_kiem", "(kiểm bằng máy: bỏ action có phủ định / khung hình; trần 3; drop_phrases phải nguyên văn trong phần mở rộng)", "",
               {"giu": sach, "bo": bo, "drop_phrases": d["drop_phrases"]})
    log(f"  [R] thiếu-prompt {len(d.get('missing_prompt_explicit') or [])} · thiếu-văn-hoá "
        f"{len(d.get('missing_cultural_identity') or [])} · action giữ {len(sach)}" + (f" · bỏ {len(bo)}" if bo else ""))
    return d


GATE_SYSTEM = (
    "You compare two blind visual reports of two images (BEFORE and AFTER a repair) against the Prompt "
    "Preservation Card and the list of repair actions that were attempted.\n"
    "fixed: attempted repairs now evidenced in AFTER but not in BEFORE.\n"
    "regressions: ONLY items that the PROMPT PRESERVATION CARD requires, that were evidenced in BEFORE, and "
    "are missing or contradicted in AFTER. The BEFORE image is NOT the standard — the prompt is. Anything in "
    "BEFORE that the card does not ask for (a wrong object, a decoration, a background detail, a pose) may "
    "change or disappear freely and is NOT a regression. Severity 'severe' only when the main entity, a "
    "stated count, or a stated supporting object from the card is lost; otherwise 'minor'.\n"
    "Be conservative about fixed: if unsure, it is not fixed. JSON only."
)
GATE_SCHEMA = {"type": "object", "properties": {
    "fixed": _arr(), "regressions": {"type": "array", "items": {"type": "object", "properties": {
        "what": {"type": "string"}, "severity": {"type": "string"}}}},
    "unchanged_issues": _arr()}, "required": ["fixed", "regressions"]}


CHECK_SYSTEM = (
    "You rate how well ONE photograph matches ONE written statement. score 0-10: 10 exactly as described, "
    "5 the thing is there but clearly differs, 0 nothing of the kind. If the part lies outside the picture set "
    "in_frame false and give no score. Never name a country or culture. evidence: what you see, max 15 words. JSON only."
)
CHECK_SCHEMA = {"type": "object", "properties": {"in_frame": {"type": "boolean"}, "score": {"type": "number"},
                "evidence": {"type": "string"}}, "required": ["in_frame"]}


def hoi_action(agent, so_tay, pid, image, action, buoc):
    """Hỏi THẲNG từng action trên một ảnh, thang 0-10 — chế độ đã đo là tin được (một ảnh + câu hỏi có đích),
    thay cho việc suy "fixed" từ bản tả mở. Ở FLUX S001 Observer tả mở khai "hai tà" cho một tấm qipao."""
    d = goi_vlm(agent, so_tay, pid, "O", buoc, CHECK_SYSTEM,
                f"Statement: \"{action}\"\nHow fully does this photograph match that statement?\n"
                'Return JSON: {"in_frame": true|false, "score": 0-10, "evidence": ".."}', CHECK_SCHEMA, image, max_new_tokens=120)
    try:
        sc = float(d.get("score")) if d.get("in_frame", True) else None
    except (TypeError, ValueError):
        sc = None
    return sc, str(d.get("evidence") or "")[:100]


def cong_gate(agent, so_tay, pid, cards, rep0, rep1, actions, log, nhan="C", i0=None, i1=None, kiem=None):
    # Chỉ đưa Preservation Card: bản đầu đưa cả report I0 làm chuẩn, cổng phạt việc BỎ ĐI chính vật sai
    # ("The cart and its contents are missing — severe" ở S002) và từ chối đúng hai tấm sửa thành công.
    user = (f"PROMPT PRESERVATION CARD (the only standard for regressions):\n"
            f"{json.dumps(cards.get('prompt_preservation'), ensure_ascii=False)}\n\n"
            f"REPAIR ACTIONS ATTEMPTED: {json.dumps(actions, ensure_ascii=False)}\n\n"
            f"BEFORE (I0) REPORT:\n{json.dumps(rep0, ensure_ascii=False)}\n\nAFTER (I1) REPORT:\n"
            f"{json.dumps(rep1, ensure_ascii=False)}\n\nReturn JSON.")
    d = goi_text(agent, so_tay, pid, "R", f"gate_{nhan}", GATE_SYSTEM, user, GATE_SCHEMA, max_new_tokens=700)
    fixed_llm = [str(x) for x in (d.get("fixed") or [])]
    # 2a. fixed tính bằng MÁY từ câu hỏi từng action trên I0 và I1: sửa được = I1 >= 7 và I0 <= 4 (hoặc I0 ngoài khung).
    fixed, bang_action = [], []
    if i0 and i1:
        for act in actions:
            s0, e0 = hoi_action(agent, so_tay, pid, i0, act, f"check_I0_{nhan}")
            s1, e1 = hoi_action(agent, so_tay, pid, i1, act, f"check_I1_{nhan}")
            ok = (s1 is not None and s1 >= 8) and (s0 is None or s0 <= 4 or s1 - s0 >= 3)
            bang_action.append({"action": act, "I0": s0, "I1": s1, "fixed": ok, "ev_I1": e1})
            if ok:
                fixed.append(act)
    else:
        fixed = fixed_llm
    regs = [r for r in (d.get("regressions") or []) if isinstance(r, dict) and str(r.get("what") or "").strip()]

    # Bốn luật máy, mỗi luật chặn một lỗi đã thấy ở lô kor_20260918_1322:
    #  1. regression trùng nội dung với một mục fixed -> mâu thuẫn, bỏ   (S002: "no longer gánh hàng rong" + fixed "Add a shoulder pole")
    #  2. chỉ so với GIÁ TRỊ của card, không so khoá JSON                 (S003: regression = "main entity", "visual effect")
    #  3. regression phải có căn cứ trong report I0 và KHÔNG còn trong report I1 (S001: "school gate is missing" khi I1 có sân trường)
    #  4. thứ không nhìn được (giờ trong ngày) không bao giờ là regression (S002: "no longer in the morning")
    def _gia_tri(o):
        if isinstance(o, dict): return " ".join(_gia_tri(v) for v in o.values())
        if isinstance(o, list): return " ".join(_gia_tri(v) for v in o)
        return str(o or "")
    card_txt = _norm(_gia_tri(cards.get("prompt_preservation")))
    txt0, txt1 = _norm(_gia_tri(rep0)), _norm(_gia_tri(rep1))
    fixed_tu = {w for f in fixed for w in _norm(f).split() if len(w) > 3}
    KHONG_NHIN = ("morning", "evening", "buổi sáng", "time of day", "noon", "afternoon", "hour")
    def _tu(s_): return [w for w in _norm(s_).split() if len(w) > 3 and w not in ("longer", "missing", "changed", "instead", "there")]
    giu, ha = [], []
    for r in regs:
        w = str(r.get("what")); tu = _tu(w); sev = _norm(r.get("severity")).startswith("sev")
        if any(k in w.lower() for k in KHONG_NHIN):
            ha.append((w, "không nhìn được")); continue
        if tu and len(set(tu) & fixed_tu) >= 2:
            ha.append((w, "mâu thuẫn với fixed")); continue
        trong_card = sum(1 for x in tu if x in card_txt)
        co_o_I0 = sum(1 for x in tu if x in txt0); con_o_I1 = sum(1 for x in tu if x in txt1)
        if sev and not (trong_card >= 1 and co_o_I0 >= 1 and con_o_I1 < co_o_I0):
            r = dict(r, severity="minor(hạ: không đủ căn cứ card/I0/I1)")
        giu.append(r)
    regs = giu
    so_tay.ghi(pid, "GATE", f"may_kiem_{nhan}", "(luật máy: bỏ regression mâu thuẫn fixed / không nhìn được; hạ severe thiếu căn cứ)", "",
               {"bo": ha, "giu": [(r.get("what"), r.get("severity")) for r in regs]})
    nang = [r for r in regs if _norm(r.get("severity")).startswith("sev")]
    # LUẬT chọn, không phải chữ của model: regression nặng -> I0; không sửa được gì -> I0; còn lại -> I1
    if nang:
        chon, ly_do = "I0", f"{len(nang)} regression nghiêm trọng"
    elif not fixed:
        chon, ly_do = "I0", "không sửa được lỗi nào"
    else:
        chon, ly_do = "I1", f"sửa được {len(fixed)}, không regression nghiêm trọng"
    # 2c. Phủ quyết ĐỘC LẬP với contract và với MLLM: I1 không được xa ảnh thật cất riêng hơn I0 quá một biên.
    veto = None
    if kiem and i0 and i1:
        sim0, sim1 = kiem(i0), kiem(i1)
        if sim0 is not None and sim1 is not None:
            veto = {"sim_I0": round(sim0, 4), "sim_I1": round(sim1, 4)}
            if sim1 < sim0 - 0.02 and chon == "I1":
                chon, ly_do = "I0", f"phủ quyết: I1 xa ảnh thật cất riêng hơn I0 ({sim1:.3f} < {sim0:.3f} - 0,02)"
    out = {"fixed": fixed, "fixed_llm": fixed_llm, "bang_action": bang_action, "regressions": regs,
           "selection": chon, "ly_do": ly_do, "veto_anh_that": veto}
    so_tay.ghi(pid, "GATE", f"luat_{nhan}", "(luật cổng, tính bằng máy)", "", out)
    log(f"  [cổng·{nhan}] fixed {len(fixed)}/{len(actions)} (LLM khai {len(fixed_llm)}) · regression {len(regs)} (nặng {len(nang)})"
        + (f" · sim thật I0 {veto['sim_I0']:.3f} → I1 {veto['sim_I1']:.3f}" if veto else "") + f" -> chọn {chon} ({ly_do})")
    return out


def dung_P1(p_ct: str, actions: list[str], card: dict | None = None, p0: str = "", drop: list[str] | None = None) -> str:
    """P1 = P_ct nguyên văn (P_orig đã nằm ở đầu) + preserve clause + <=3 action dương tính. Không negative.

    Preserve clause nay LIỆT KÊ vật phụ và bối cảnh của Preservation Card. Đo ở kor_20260918_1349: IP-Adapter
    sửa đúng định danh (áo dài, bánh chưng) nhưng kéo ảnh về bố cục ảnh thật, làm rơi "cổng trường" và
    "mâm" mà prompt yêu cầu, rồi cổng từ chối đúng hai tấm đó. Nêu tên vật phải giữ là cách rẻ nhất.
    """
    if not actions:
        return p_ct
    # Phẫu thuật phần mở rộng: P0 giữ nguyên văn ở đầu, chỉ cắt cụm trong phần Culture-TRIP viết thêm.
    if p0 and p_ct.startswith(p0) and drop:
        mo_rong = p_ct[len(p0):]
        for x in drop:
            i = mo_rong.lower().find(x.lower())
            if i >= 0:
                mo_rong = mo_rong[:i] + mo_rong[i + len(x):]
        p_ct = p0 + " " + " ".join(mo_rong.replace(" ,", ",").replace(" .", ".").split())
    ds = "\n".join(f"{i + 1}. {a.rstrip('.')}." for i, a in enumerate(actions))
    giu = []
    for k in ("supporting_objects", "background_and_scene"):
        for x in ((card or {}).get(k) or []):
            x = str(x).strip()
            if x and re.search(r"[A-Za-z]", x) and not re.search(r"[ăâđêôơưàáảãạ]", x.lower()):
                giu.append(x)                      # chỉ lấy bản tiếng Anh; SDXL không đọc tiếng Việt
    keep = f" Keep clearly visible: {'; '.join(dict.fromkeys(giu))}." if giu else ""
    return (f"{p_ct.strip()}\n\nPreserve the current subject, composition, setting, colors, and all correctly "
            f"rendered details.{keep} Make these additions clearly visible:\n{ds}")

# ------------------------------------------------------------------ lưới + transcript
def _font(size, bold=False):
    from PIL import ImageFont
    n = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    p = f"/usr/share/fonts/truetype/dejavu/{n}"
    return ImageFont.truetype(p, size) if Path(p).exists() else ImageFont.load_default()


def ve_luoi(hang, cot, out_png, cell=300, nhan=None, ten_hang=None):
    from PIL import Image, ImageDraw
    import textwrap
    gut, head, pad, cap = 210, 30, 6, 22
    cw, ch = cell + pad, cell + cap + pad
    W, H = gut + len(cot) * cw + pad, head + len(hang) * ch + pad
    im = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(im)
    for i, c in enumerate(cot):
        d.text((gut + i * cw + 3, 8), (nhan or {}).get(c, c), font=_font(14, True), fill=(20, 20, 20))
    for r, (pid, o, ghi) in enumerate(hang):
        y = head + r * ch
        nh = (ten_hang or {}).get(pid, pid)
        for j, dong in enumerate(textwrap.wrap(nh, 30)[:9]):
            d.text((4, y + 6 + j * 15), dong, font=_font(12, j == 0), fill=(20, 20, 20))
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
    ap.add_argument("--run-name", default=None, help="mặc định kor_<YYYYmmdd_HHMM> để không ghi đè lô cũ")
    ap.add_argument("--append-run", default=None, help="nối prompt mới vào lô đã có (đường dẫn thư mục run); lưới vẽ lại gồm cả cũ")
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--strength", type=float, default=0.60)   # 0.35 giữ bố cục tốt tới mức không đổi được vật thể
    ap.add_argument("--wiki", default=str(ROOT / "data" / "wiki_curated" / "S001_S003.json"))
    ap.add_argument("--i2i", action="store_true", help="thêm cột tham khảo C-i2i (img2img từ I0, không ref)")
    ap.add_argument("--c-text", action="store_true", help="thêm cột C-text: cùng seed + P1, KHÔNG adapter (tách chữ khỏi ref)")
    ap.add_argument("--method-name", default="CG-MAPR", help="tên phương pháp in trên lưới (Contract-Guided Multi-Agent Prompt Repair)")
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args(argv)

    ov: dict = {}
    set_dotted(ov, "t2i.render", "bare"); set_dotted(ov, "multigen.adaptive.enabled", "false")
    set_dotted(ov, "multigen.n_candidates", "1"); set_dotted(ov, "multigen.keep_loaded", "1")
    for kv in a.set:                       # --set của người dùng ghi đè mặc định (FLUX cần cpu_offload, keep_loaded=0)
        k, _, v = kv.partition("="); set_dotted(ov, k, v)
    cfg = Config.load(a.config, ov)

    wiki = json.loads(Path(a.wiki).read_text(encoding="utf-8"))
    allp = {p.id: p for p in load_prompts(cfg.prompts_path)}
    ids = [i.strip() for i in a.ids.split(",") if i.strip() in allp]
    if a.append_run:
        run_dir = Path(a.append_run)
    else:
        run_dir = Path(cfg.runs_dir) / (a.run_name or time.strftime("kor_%Y%m%d_%H%M"))
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run: {run_dir}", flush=True)
    log = lambda *x: print(*x, flush=True)  # noqa: E731
    so_tay = SoTay()

    from ctig.models import loader as ML, registry as REG
    from ctig.stages import multigen as mg
    from ctig.stages.generation import DiffusersGenerator
    REGISTRY = getattr(REG, "MODELS", None) or getattr(REG, "REGISTRY")

    # C = cùng seed + IP-Adapter ảnh thật + P1. Đo hai lô: img2img 0.35 và 0.60 đều không đổi được vật thể
    # sai (xe đẩy vẫn xe đẩy, hoa vẫn hoa), còn IP-Adapter sửa đúng 2/2 (S001 áo dài, S002 gánh hàng rong).
    cot = ["A", "B (I0)", "C (I1)"] + (["C-text"] if a.c_text else []) + (["C-i2i"] if a.i2i else [])
    M = {"sdxl_base": "SDXL", "realvis_xl": "RealVisXL", "flux_dev": "FLUX.1-dev"}.get(a.model.split("#")[0].split("+")[0], a.model)
    NHAN = {"A": M, "B (I0)": f"{M} + Culture-TRIP", "C (I1)": f"{M} + {a.method_name}",
            "C-text": f"{M} + {a.method_name} (text-only)", "C-i2i": f"{M} + img2img (đối chứng)"}
    hang, tong = [], []
    so_tay = so_tay
    cu_json = run_dir / "kor.json"
    if a.append_run and cu_json.exists():        # nạp lô cũ để lưới/transcript gồm cả cũ lẫn mới
        cu = json.loads(cu_json.read_text(encoding="utf-8"))
        tong = cu.get("don_vi", []); so_tay.dong = cu.get("giao_tiep", [])
        for u in tong:
            hang.append((u["prompt_id"], u["images"], {k: ("= I0" if k == "B (I0)" else "prompt gốc" if k == "A" else
                        (f"chọn {(u.get('gate_C') or {}).get('selection')}" if k == "C (I1)" else "")) for k in u["images"]}))
        da_co = {u["prompt_id"] for u in tong}
        ids = [i for i in ids if i not in da_co]
        print(f"nối vào lô cũ: đã có {sorted(da_co)} · chạy thêm {ids}", flush=True)
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
        dem = {"A": 0, "B": 0, "C": 0, "C-text": 0, "C-i2i": 0}

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
            dem["C-i2i"] += 1
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
        tu_khoa = list(getattr(pr, "gold_entities", None) or []) + [w for w in pr.text_vi.split() if len(w) > 3]
        try:   # danh từ trong contract v2 của team (entity_vi, part, mô tả) làm mỏ neo cắt bài Wikipedia
            cv = json.loads((ROOT / "data" / "contracts_v2.json").read_text(encoding="utf-8")).get(pid, {})
            tu_khoa += [cv.get("entity_vi", "")] + [str(r.get("part", "")).replace("|", " ") for r in cv.get("required", [])]
            tu_khoa += [w for r in cv.get("required", []) for w in str(r.get("description", "")).split() if len(w) > 5]
        except Exception:  # noqa: BLE001
            pass
        cards = agent_K(s.agent, so_tay, pid, pr.text_vi, p_orig, wiki.get(pid, []), log, tu_khoa=tu_khoa)
        # ---- O: quan sát mù I0
        rep0 = agent_O(s.agent, so_tay, pid, i0, "I0", log)
        # ---- R: phân tích lỗ hổng -> action
        mo_rong = p_ct[len(p_orig):].strip() if p_ct.startswith(p_orig) else ""
        gap = agent_R(s.agent, so_tay, pid, p_orig, cards, rep0, log, mo_rong=mo_rong)
        actions = gap.get("repair_actions") or []
        p1 = dung_P1(p_ct, actions, cards.get("prompt_preservation"), p0=p_orig, drop=gap.get("drop_phrases"))
        if gap.get("drop_phrases"):
            log(f"  [P1] cắt khỏi phần mở rộng: {gap['drop_phrases']}")
        so_tay.ghi(pid, "P1", "prompt", "(P1 = P_ct nguyên văn + preserve clause + action; không negative)", "", {"P1": p1})

        _, eval_refs = ref_split(cfg.retrieval.ref_dir, pid, 5)      # candidates/: chỉ để chấm, IP-Adapter không nhìn
        def kiem(img):
            return ref_similarity(s.clip, img, eval_refs)
        anh = {"A": anh_A, "B (I0)": i0}
        ghi = {"B (I0)": "= I0", "A": "prompt gốc"}
        gate_C = gate_i2i = gate_text = None
        if not actions:
            log("  [C] no-op: R không có action hợp lệ -> I1 = I0")
            anh["C (I1)"] = i0; ghi["C (I1)"] = "no-op (= I0)"
            if a.i2i:
                anh["C-i2i"] = i0; ghi["C-i2i"] = "no-op (= I0)"
        else:
            # ---- C: cùng seed, text2img, IP-Adapter trên ảnh thật đã cắt về chủ thể, prompt P1
            refs, _ = ref_split(cfg.retrieval.ref_dir, pid, 5)
            refs = refs[:cfg.multigen.ref_images]
            if refs and cfg.multigen.ref_crop:
                refs = s.crop_refs(refs, s.spec()[0])
            i1 = sinh(p1, "C_ref", "C", refs=refs) if refs else None
            log(f"  [C] cùng seed + IP-Adapter {len(refs)} ảnh thật + P1 -> {i1}")
            try:   # kiểm bằng máy: multigen.json của hàng C phải có ghi chú IP-Adapter, không thì ref đã bị bỏ im lặng
                mj = json.loads((out_dir / "C_ref" / "multigen.json").read_text(encoding="utf-8"))
                notes = " ".join(n for r in mj.get("runs", []) for n in (r.get("notes") or []))
                if "IP-Adapter" not in notes:
                    log("  [!] C: KHÔNG thấy IP-Adapter trong ghi chú multigen -> ảnh thật không được gắn (khoá '#bare' xoá cờ +ref?)")
            except Exception:  # noqa: BLE001
                pass
            rep1 = agent_O(s.agent, so_tay, pid, i1, "I1", log) if i1 else {}
            gate_C = cong_gate(s.agent, so_tay, pid, cards, rep0, rep1, actions, log, "C", i0=i0, i1=i1, kiem=kiem) if i1 else None
            anh["C (I1)"] = i1 or i0
            ghi["C (I1)"] = f"chọn {gate_C['selection']}" if gate_C else ("không có ảnh thật" if not refs else "sinh hỏng")
            # ---- C-text (tuỳ chọn): cùng seed + P1, KHÔNG adapter — tách "chữ có đủ không" khỏi "adapter có phá không"
            gate_text = None
            if a.c_text:
                i1t = sinh(p1, "C_text", "C-text")
                log(f"  [C-text] cùng seed + P1, không adapter -> {i1t}")
                rep1t = agent_O(s.agent, so_tay, pid, i1t, "I1text", log) if i1t else {}
                gate_text = cong_gate(s.agent, so_tay, pid, cards, rep0, rep1t, actions, log, "text",
                                      i0=i0, i1=i1t, kiem=kiem) if i1t else None
                anh["C-text"] = i1t or i0
                ghi["C-text"] = f"chọn {gate_text['selection']}" if gate_text else "sinh hỏng"
            # ---- C-i2i (tuỳ chọn): img2img từ I0, để đối chứng
            if a.i2i:
                i1b = sinh_i2i(p1, i0, "C_i2i")
                log(f"  [C-i2i] img2img strength {a.strength} -> {i1b}")
                rep1b = agent_O(s.agent, so_tay, pid, i1b, "I1i2i", log) if i1b else {}
                gate_i2i = cong_gate(s.agent, so_tay, pid, cards, rep0, rep1b, actions, log, "i2i", i0=i0, i1=i1b, kiem=kiem) if i1b else None
                anh["C-i2i"] = i1b or i0
                ghi["C-i2i"] = f"chọn {gate_i2i['selection']}" if gate_i2i else "sinh hỏng"

        rec = {"prompt_id": pid, "prompt_vi": pr.text_vi, "P_orig": p_orig, "P_ct": p_ct, "P1": p1, "seed": a.seed,
               "model": a.model, "strength_i2i": a.strength, "images": anh, "so_lan_sinh": dem,
               "cards": cards, "report_I0": rep0, "gap": gap, "actions": actions,
               "gate_C": gate_C, "gate_text": gate_text, "gate_i2i": gate_i2i, "refs": refs if actions else [],
               "drop_phrases": gap.get("drop_phrases") or []}
        (out_dir / "kor.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        tong.append(rec); hang.append((pid, anh, ghi))
        ve_luoi(hang, cot, run_dir / "kor_grid.png", nhan=NHAN, ten_hang={u["prompt_id"]: f"{u['prompt_id']}: {u['prompt_vi']}" for u in tong})
        (run_dir / "kor.json").write_text(json.dumps({"don_vi": tong, "giao_tiep": so_tay.dong},
                                                     ensure_ascii=False, indent=1), encoding="utf-8")
        transcript_md(so_tay, run_dir / "kor_transcript.md")
        log(f"  sinh: {dem} · -> {out_dir / 'kor.json'}")

    log("\nKOR_DONE")


if __name__ == "__main__":
    main()
