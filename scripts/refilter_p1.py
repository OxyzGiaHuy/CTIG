"""Lọc máy đầu ra của R đã lưu (actions, drop_phrases) theo 4 luật, dựng lại P1 cho TOÀN BỘ đơn vị, ghi kế hoạch sinh lại.

Luật (áp cho mọi đơn vị, không nhìn metric):
  R1 neo   : action phải bám vật thể có trong P0 / identity cue / disambiguator / gap (missing_*, contradictions).
             Neo tính trên phần đầu của action (trước " on | in | next to | at | around | under | over | with " ...),
             để "a spoon next to the chopsticks on the wooden table" không được neo nhờ "wooden table".
             KHÔNG neo vào conditional_cues và supporting_objects (đã nằm trong Keep clause).
  R2 câu   : bỏ action dạng định nghĩa ("X is a ..."), > 18 từ, hoặc có số đo (cm/m).
  R3 lặp   : cắt đuôi ≥4 từ lặp ở ≥2 action; bỏ trùng; trần 3.
  R4 drop  : chỉ giữ drop_phrase có dấu hiệu ngoại lai/so sánh (Chinese, Taj Mahal, comparable to ...) hoặc nằm trong
             gap.contradictions; cấm drop cụm có dấu tiếng Việt (khăn xếp) — đó là tên thực thể.
  No-op    : không action, không drop -> P1 = P_ct nhưng VẪN sinh với IP-Adapter (trước đây rơi về I0, không ref).

    python scripts/refilter_p1.py --run docs/report_assets/kor_20260918_1349 -o docs/report_assets/plan_sdxl.json
"""
import argparse, json, re
from pathlib import Path

STOP = set("a an the of in on at to with and or for from by into onto over under next around near front back top center centre "
           "clearly visible small large big little one two three some several many its his her their there here where while as is are "
           "made make making set placed placing resting standing sitting hanging holding held table hand hands people person man woman "
           "scene image background foreground side sides part parts piece".split())
FOREIGN = re.compile(r"\b(chin(a|ese)|japan(ese)?|korea(n)?|thai(land)?|india(n)?|taj mahal|borneo|malay|indonesia|western|europe(an)?|"
                     r"american|french|opera|kimono|hanfu|qipao|cheongsam|sari|similar to|comparable to|reminiscent|akin to|resembl\w*|like (a|an|the))\b", re.I)
VI = re.compile(r"[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]", re.I)
SPLIT = re.compile(r"\b(on|in|next to|at|around|under|over|beside|behind|inside|within|along|across|among|near|from|for|where|with|under)\b", re.I)

def stems(t): return {w[:5] for w in re.findall(r"[a-zA-Z]{4,}", t.lower()) if w not in STOP}
def head(a): a = re.sub(r"^clearly visible:\s*", "", a.strip(), flags=re.I); return SPLIT.split(a, 1)[0]

def anchors(u):
    g = u.get("gap") or {}; ce = (u.get("cards") or {}).get("cultural_evidence") or {}
    t = [u.get("P_orig", "")] + list(g.get("missing_prompt_explicit") or []) + list(g.get("missing_cultural_identity") or []) + list(g.get("contradictions") or [])
    for c in ce.get("identity_cues") or []: t += [str(c.get("cue_en") or ""), str(c.get("visual_check_en") or "")]
    for c in ce.get("positive_disambiguators") or []:
        if isinstance(c, dict): t += [str(c.get("cue_en") or ""), str(c.get("target_appearance") or "")]
    return stems(" ".join(t))

def filter_actions(u):
    A = anchors(u); out, why = [], []
    acts = [a.strip() for a in (u.get("actions") or []) if a.strip()]
    for a in acts:
        core = re.sub(r"^clearly visible:\s*", "", a, flags=re.I)
        if re.search(r"\b(is|are|was|were)\s+(a|an|the|mainly|only|often|typically|usually)\b", core, re.I) or len(core.split()) > 18 or re.search(r"\d+\s?(cm|mm|m|km)\b", core):
            why.append(f"R2 bỏ: {a}"); continue
        if not (stems(head(a)) & A):
            why.append(f"R1 bỏ (không neo '{head(a).strip()}'): {a}"); continue
        out.append(a)
    # R3: đuôi lặp
    if len(out) >= 2:
        tails = {}
        for a in out:
            w = a.split()
            for n in range(4, min(8, len(w))):
                tails.setdefault(" ".join(w[-n:]).lower(), set()).add(a)
        rep = [t for t, s in tails.items() if len(s) >= 2]
        if rep:
            t = max(rep, key=len); out = [re.sub(re.escape(t) + r"$", "", a, flags=re.I).strip() or a for a in out]; why.append(f"R3 cắt đuôi lặp '{t}'")
    dedup = list(dict.fromkeys(out))
    if len(dedup) < len(out): why.append("R3 bỏ trùng")
    return dedup[:3], why

def filter_drops(u):
    g = u.get("gap") or {}; contra = " ".join(g.get("contradictions") or []).lower(); out, why = [], []
    for d in (u.get("drop_phrases") or g.get("drop_phrases") or []):
        d = str(d).strip()
        if not d: continue
        if VI.search(d): why.append(f"R4 giữ lại cụm tiếng Việt: {d}"); continue
        if FOREIGN.search(d) or d.lower() in contra: out.append(d)
        else: why.append(f"R4 không drop (không ngoại lai): {d}")
    return out, why

def build_P1(p_ct, p0, actions, drop, card):
    if p0 and p_ct.startswith(p0) and drop:
        mo = p_ct[len(p0):]
        for x in drop:
            i = mo.lower().find(x.lower())
            if i >= 0: mo = mo[:i] + mo[i + len(x):]
        p_ct = p0 + " " + " ".join(mo.replace(" ,", ",").replace(" .", ".").replace(",,", ",").split())
    if not actions: return p_ct.strip()
    giu = [str(x).strip() for k in ("supporting_objects", "background_and_scene") for x in ((card or {}).get(k) or [])
           if re.search(r"[A-Za-z]", str(x)) and not VI.search(str(x))]
    keep = f" Keep clearly visible: {'; '.join(dict.fromkeys(giu))}." if giu else ""
    ds = "\n".join(f"{i + 1}. {a.rstrip('.')}." for i, a in enumerate(actions))
    return (f"{p_ct.strip()}\n\nPreserve the current subject, composition, setting, colors, and all correctly rendered details.{keep} "
            f"Make these additions clearly visible:\n{ds}")

ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("-o", required=True); a = ap.parse_args()
rd = Path(a.run); units = json.loads((rd / "kor.json").read_text(encoding="utf-8"))["don_vi"]; plan = {}; n_chg = 0
for u in units:
    if "ablation" in u: continue
    acts, w1 = filter_actions(u); drops, w2 = filter_drops(u)
    p1 = build_P1(u["P_ct"], u["P_orig"], acts, drops, (u.get("cards") or {}).get("prompt_preservation"))
    i1_la_i0 = u["images"].get("C (I1)") == u["images"].get("B (I0)")
    changed = (p1.strip() != (u.get("P1") or "").strip()) or i1_la_i0
    if i1_la_i0: w1.append("no-op cũ rơi về I0 (không ref) -> nay sinh với ref")
    if changed and not (w1 + w2): w1.append("R5 Keep clause: bỏ mục tiếng Việt lọt qua (SDXL/FLUX không đọc)")
    plan[u["prompt_id"]] = {"changed": changed, "actions_old": u.get("actions") or [], "actions_new": acts, "drop_old": u.get("drop_phrases") or [], "drop_new": drops,
                            "P1_old": u.get("P1"), "P1_new": p1, "ly_do": w1 + w2, "seed": u["seed"]}
    n_chg += changed
Path(a.o).write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"{rd.name}: {n_chg}/{len(plan)} đơn vị đổi P1 -> {a.o}")
for pid, p in plan.items():
    if p["changed"]: print(f"  {pid}: {p['actions_old']} -> {p['actions_new']} | drop {p['drop_old']} -> {p['drop_new']}\n      " + "\n      ".join(p["ly_do"]))
