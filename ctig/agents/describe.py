"""
Filter agent (v1.4): hai tầng, không hỏi có/không trên ảnh.

  tầng 1  VLM mô tả ảnh có cấu trúc (số người, chủ thể, trang phục, vật, nền)      -> ImageDescriptor
  tầng 2  so MÔ TẢ (văn bản) với must_have_en / must_not_en: agent văn bản trả present/absent/unsure
          kèm cụm trích từ mô tả; cụm trích không có trong mô tả -> hạ xuống unsure   -> FilterVerdict

Luật giữ/bỏ (rẻ, kiểm được, không phải LLM):
  reference: bỏ nếu (prompt một người mà ảnh >= 3 người) hoặc (không có must_have nào mà có must_not);
  candidate: bỏ nếu có must_not, hoặc sai số người khi prompt nói rõ một người.
  Luôn giữ ít nhất một ảnh (điểm cao nhất) để bước sau không rỗng.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..schema import CulturalSpec, FilterResult, FilterVerdict, ImageDescriptor

_SINGULAR = re.compile(r"\b(a|an|one|single)\s+(young\s+|old\s+|little\s+|elderly\s+)?(woman|girl|man|boy|person|lady|child|fisherman|farmer|vendor|monk|calligrapher|dancer|musician)\b", re.I)
_PLURAL = re.compile(r"\b(two|three|four|several|group|crowd|people|women|men|girls|boys|children|family|dancers|musicians|students|villagers|couple)\b", re.I)


def expected_people(prompt_en: str) -> int | None:
    """1 nếu prompt nói rõ một người và không có từ số nhiều; None nếu không ràng buộc."""
    if not prompt_en:
        return None
    if _PLURAL.search(prompt_en):
        return None
    return 1 if _SINGULAR.search(prompt_en) else None


def describe(agent, paths: list[str], log=print) -> list[ImageDescriptor]:
    out = []
    for p in paths:
        try:
            d = agent.describe_image(p)
        except Exception as exc:  # noqa: BLE001
            log(f"  [filter] mô tả {p}: {type(exc).__name__}: {str(exc)[:80]}")
            d = {}
        pc = d.get("people_count")
        try:
            pc = int(pc) if pc is not None else None
        except (TypeError, ValueError):
            pc = None
        out.append(ImageDescriptor(
            path=p, people_count=pc,
            subjects=[str(x) for x in d.get("subjects", []) if x][:6],
            garments=[str(x) for x in d.get("garments", []) if x][:8],
            objects=[str(x) for x in d.get("objects", []) if x][:8],
            background=str(d.get("background", ""))[:200],
            watermark_or_text=bool(d.get("watermark_or_text", False)),
        ))
    return out


def _garment_fields(desc: ImageDescriptor) -> list[dict]:
    """VLM hay trả mỗi garment là một dict (dạng chuỗi). Đọc lại thành dict; chuỗi thường thì gói vào {'text': ...}."""
    import ast

    out = []
    for g in desc.garments:
        d = None
        if isinstance(g, dict):
            d = g
        else:
            s = str(g).strip()
            if s.startswith("{"):
                try:
                    d = ast.literal_eval(s)
                except Exception:  # noqa: BLE001
                    d = None
        out.append({k: str(v).lower() for k, v in d.items()} if isinstance(d, dict) else {"text": s.lower()})
    return out


_NONE = {"", "none", "no", "null", "n/a", "not visible", "unknown"}


def garment_rules(desc: ImageDescriptor, attr: str) -> str | None:
    """Luật cứng trên các trường trang phục VLM đã mô tả (collar / lower_body / slits / sash_or_belt / type).
    Trả 'present' | 'absent' | None (không có luật -> để agent văn bản quyết). v1.4 p001: agent văn bản tính
    'high stand-up collar' là có trong khi mô tả ghi collar: crossed, lower_body: not visible."""
    a = attr.lower()
    gs = _garment_fields(desc)
    if not gs:
        return None

    def vals(field):
        return [g.get(field, "") for g in gs if g.get(field, "") not in _NONE]

    if "collar" in a:
        cs = vals("collar")
        if not cs:
            return None
        if any(k in a for k in ("stand", "mandarin", "high")):
            return "present" if any(("stand" in c or "mandarin" in c or "high" in c) for c in cs) else "absent"
        if "cross" in a or "y-shaped" in a or "v-neck" in a:
            return "present" if any(("cross" in c or "v" == c[:1] or "y" in c) for c in cs) else "absent"
    if "trouser" in a or "pants" in a or "bare legs" in a or "no trousers" in a:
        lb = vals("lower_body")
        if not lb:
            return None
        has_trousers = any(("trouser" in v or "pant" in v) for v in lb)
        bare = any(("bare" in v or "skirt" in v or "dress" in v or "gown" in v) for v in lb)
        if "no trousers" in a or "bare legs" in a or "without trousers" in a:
            return "present" if (bare and not has_trousers) else "absent"
        return "present" if has_trousers else ("absent" if bare else None)
    if "slit" in a or "split" in a:
        sl = vals("slits")
        if not sl:
            return None
        return "present" if any(v not in ("no", "none") for v in sl) else "absent"
    if "obi" in a or "sash" in a or "belt" in a:
        # v1.9: phân biệt "mô tả KHÔNG nói gì về đai" (-> None, để VQA/agent quyết) với "mô tả nói không có đai" (-> absent).
        # Trước đây cả hai đều ra 'absent' vì vals() lọc mất giá trị 'none' -> S031: 24/32 ảnh bị ghi thiếu 'silk sash'
        # dù VQA trung vị 0,94.
        if not any("sash_or_belt" in g for g in gs):
            return None
        sb = vals("sash_or_belt")
        return "present" if any(v not in ("no", "none") for v in sb) else "absent"
    if "one-piece" in a or "gown" in a or "floor-length" in a:
        ty = vals("type"); ln = vals("length"); lb = vals("lower_body")
        if any(("dress" in v or "gown" in v) for v in ty) and not any(("trouser" in v or "pant" in v) for v in lb):
            return "present"
        if any(("trouser" in v or "pant" in v) for v in lb):
            return "absent"
    return None


def compact_text(desc: ImageDescriptor, max_chars: int = 700) -> str:
    """Mô tả gọn cho agent văn bản: dict trang phục -> 'type=dress, fit=loose, ...'; cắt ở max_chars.
    (Mô tả thô có dict dài -> Qwen chép nguyên vào JSON -> vượt max_new_tokens -> không parse được.)"""
    parts = [f"people: {desc.people_count}"] if desc.people_count is not None else []
    if desc.subjects:
        parts.append("subjects: " + "; ".join(desc.subjects[:4]))
    for i, g in enumerate(_garment_fields(desc)[:3]):
        if "text" in g:
            parts.append(f"garment{i + 1}: {g['text'][:120]}")
        else:
            parts.append(f"garment{i + 1}: " + ", ".join(f"{k}={v}" for k, v in g.items() if v and v not in _NONE)[:160])
    if desc.objects:
        parts.append("objects: " + "; ".join(str(o)[:60] for o in desc.objects[:4]))
    if desc.background:
        parts.append("background: " + desc.background[:120])
    return " | ".join(parts)[:max_chars]


_PAIR_NOUNS = ("collar", "trousers", "pants", "sash", "obi", "skirt", "slit", "sleeve", "hat", "brim", "broth", "noodle")

#: Nhóm vùng nhìn thấy được, để ghép một must_have với must_not NÓI VỀ CÙNG CHỖ (v1.9.2). Cặp này thành hai lựa chọn
#: loại trừ nhau của câu trắc nghiệm. Ghép bằng luật thay vì hỏi LLM: LLM viết "short-sleeved tunic without any splits"
#: cho thuộc tính "tunic split at the hips into front and back panels" - lật nhầm chiều dài tay áo, ảnh áo liền quần
#: vẫn được 0,97. must_not viết tay/rút từ web đã sẵn là mô tả sai đúng chỗ.
_REGION_GROUPS: dict[str, tuple[str, ...]] = {
    "silhouette": ("split", "panels", "panel", "flap", "two-piece", "tunic", "one-piece", "onepiece", "jumpsuit",
                   "gown", "joined", "single piece", "no trousers", "seamless"),
    "collar": ("collar", "neckline", "neck", "lapel"),
    "lower": ("trousers", "pants", "skirt", "legs", "hem", "bare legs", "shorts"),
    "waist": ("sash", "obi", "belt", "waistband", "girdle"),
    "sleeve": ("sleeve", "cuff", "arm"),
    "head": ("hat", "brim", "conical", "headdress", "turban", "crown"),
    "material": ("embroider", "brocade", "silk", "lace", "print", "pattern"),
}


def _groups(text: str) -> set[str]:
    t = (text or "").lower()
    return {g for g, words in _REGION_GROUPS.items() if any(w in t for w in words)}


_NEGATORS = ("without", "no", "not", "lacking", "missing", "absent", "lacks", "none")
_STOPW = {"any", "a", "an", "the", "of", "at", "in", "on", "with", "and", "or", "its", "their", "that", "which"}


def usable_distractor(attr: str, alt: str) -> bool:
    """Mô tả sai chỉ dùng được nếu nó nêu một HÌNH DẠNG KHÁC, không phải chính thuộc tính bị phủ định.

    "short-sleeved tunic without any splits" cho thuộc tính "tunic split at the hips into front and back panels"
    chỉ phủ định chữ 'split' -> model chọn A vì tay áo đúng là dài, ảnh áo liền quần vẫn được 0,97. Ngược lại
    "one-piece dress with no trousers underneath" nêu một dáng khác hẳn -> ảnh sai rớt còn 0,15.
    """
    alt = (alt or "").strip()
    if not alt or not (3 <= len(alt.split()) <= 20) or alt.lower() == attr.lower():
        return False
    aw = {w.strip(".,;:\"'()") for w in attr.lower().split()}
    toks = [w.strip(".,;:\"'()") for w in alt.lower().split()]
    for i, w in enumerate(toks):
        if w not in _NEGATORS:
            continue
        for nxt in toks[i + 1:i + 4]:           # chữ bị phủ định nằm ngay sau từ phủ định
            if nxt in _STOPW or not nxt:
                continue
            if nxt in aw or any(nxt.startswith(a[:5]) and len(a) >= 5 for a in aw):
                return False                     # chỉ là phủ định chính thuộc tính
            break
    ga, gl = _groups(attr), _groups(alt)
    return not ga or bool(ga & gl)


def pair_distractors(have_all: list[str], not_all: list[str]) -> dict[str, str]:
    """Ghép mỗi must_have với must_not cùng vùng để làm lựa chọn còn lại của câu trắc nghiệm.

    Một must_not được dùng cho nhiều must_have nếu hợp. Khi nhiều must_not cùng khớp, chọn cái HẸP hơn
    (ít nhóm hơn) vì nó nói đúng một chỗ: "puffy flared skirt" tốt hơn "one-piece dress with no trousers"
    khi đối chiếu với "worn over wide-legged long trousers".
    """
    out: dict[str, str] = {}
    cand = [(n, _groups(n)) for n in not_all if n]
    for h in have_all:
        gh = _groups(h)
        if not gh:
            continue
        best, best_key = None, None
        for n, gn in cand:
            shared = gh & gn
            if not shared or not usable_distractor(h, n):
                continue
            key = (len(shared), -len(gn))  # nhiều nhóm chung trước, rồi must_not hẹp hơn
            if best_key is None or key > best_key:
                best, best_key = n, key
        if best:
            out[h] = best
    return out



def _counterpart(not_attr: str, have_attrs: list[str]) -> str | None:
    """must_have nói về cùng bộ phận với must_not (cổ áo, phần dưới, đai...) để CLIP so cặp."""
    a = not_attr.lower()
    nouns = [n for n in _PAIR_NOUNS if n in a]
    if "no trousers" in a or "bare legs" in a or "gown" in a or "one-piece" in a:
        nouns += ["trousers"]
    for h in have_attrs:
        hl = h.lower()
        if any(n in hl for n in nouns):
            return h
    return None


def attr_question_neg(name_en: str, attr_en: str) -> str:
    """Câu PHỦ ĐỊNH đối chứng cho cùng thuộc tính (v1.9.1). VLM có thiên lệch GẬT: hỏi 'áo có tà xẻ không' thì gật cả với ảnh
    áo liền quần (S001 sdxl_base c1 được +1,00 dù không có tà). Hỏi thêm câu ngược lại: nếu model gật CẢ HAI thì nó không thật
    sự phân biệt được -> coi là KHÔNG ĐO ĐƯỢC, không tính là có."""
    return (f'Look carefully at the {name_en} in this photo. Statement: "the {name_en} does NOT have {attr_en}". '
            "Is this statement true for what you see? Answer Yes or No.")


def attr_question(name_en: str, attr_en: str) -> str:
    """Câu hỏi VQA dạng PHÁT BIỂU (v1.8.1). Trước đây ghép "Does the ao dai have worn over wide-legged trousers?" -> sai ngữ pháp,
    Qwen trả No cho cả ảnh áo dài thật (điểm kiểm 0,0 cho thuộc tính đúng). Dạng phát biểu tách thuộc tính khỏi cấu trúc câu."""
    return (f'Look carefully at the {name_en} in this photo. Statement: "{attr_en}". '
            "Is this statement true for what you see? Answer Yes or No.")


def attr_choice_question(name_en: str, opt_a: str, opt_b: str) -> str:
    """Câu TRẮC NGHIỆM hai lựa chọn loại trừ nhau + lối thoát 'không cái nào' (v1.9.2)."""
    cap = lambda t: t[0].upper() + t[1:] if t else t
    return (f"Look at the {name_en} in this photo. Which ONE of these matches what you actually see?\n"
            f"A. {cap(opt_a)}\nB. {cap(opt_b)}\nC. neither A nor B\n"
            "Answer with only the letter A, B or C.")


def forced_choice(agent, name_en: str, path: str, have_attr: str, alt_attr: str) -> float | None:
    """P(thuộc tính đúng) qua câu trắc nghiệm, chạy CẢ HAI thứ tự rồi lấy trung bình để khử thiên lệch vị trí.

    Đo trên S001 (Qwen2.5-VL-7B), thuộc tính "tunic split at the hips into front and back panels":
    câu có/không cho 0,87 với ảnh áo liền quần và 0,90-0,96 với ba ảnh áo dài thật -> không tách được.
    Câu trắc nghiệm cho 0,05/0,24 với ảnh sai và 0,84-0,96 với ba ảnh đúng.
    """
    fn = getattr(agent, "vqa_choice", None)
    if fn is None or not alt_attr:
        return None
    p1 = fn(attr_choice_question(name_en, have_attr, alt_attr), path)
    p2 = fn(attr_choice_question(name_en, alt_attr, have_attr), path)
    if p1 is None or p2 is None:
        return None
    return (p1[0] + p2[1]) / 2.0


def clip_agrees_not(clip, path: str, name_en: str, not_attr: str, have_attr: str, margin: float = 0.60) -> bool | None:
    """CLIP so cặp trên chính ảnh: P('với must_not') so với P('với must_have'). True = CLIP cũng thấy must_not;
    False = CLIP nghiêng về must_have (VLM đọc sai); None = không kiểm được."""
    if clip is None or not hasattr(clip, "probs"):
        return None
    try:
        p = clip.probs(path, [f"a photo of a {name_en} with {have_attr}", f"a photo of a {name_en} with {not_attr}"])
        return p[1] >= (1.0 - margin)  # must_not phải chiếm >= 40% mới coi là CLIP đồng ý
    except Exception:  # noqa: BLE001
        return None


def _verdict(agent, desc: ImageDescriptor, spec: CulturalSpec, kind: str, n_people: int | None, clip=None,
             alts: dict[str, str] | None = None, calib: dict[str, dict] | None = None) -> FilterVerdict:
    have_all: list[str] = []
    not_all: list[str] = []
    # v1.6 vast p050: prompt chỉ có thực thể bối cảnh (Tết) -> 0/0 thuộc tính, Filter mù. Thực thể context cũng có must_have
    # (cây quất, bao lì xì) / must_not (lồng đèn Trung Quốc) kiểm được trên mô tả objects/background -> đưa vào, vật thể trước.
    key_attrs: set[str] = set()
    # chỉ thực thể đủ trọng số (nêu tên hoặc suy ra chắc); thực thể đoán w<0.6 (C008: áo dài 0.55 trong prompt áo tứ thân)
    # không được áp must_have lên ảnh
    for se in sorted([s for s in spec.entities if s.weight >= 0.6] or spec.entities, key=lambda s: 0 if s.kind == "object" else 1):
        attrs = [a for a in se.required_attrs_en if a]
        have_all += attrs
        key_attrs.update(attrs[:2])  # KB viết tay xếp 2 thuộc tính ĐỊNH DANH (cổ đứng, hai tà...) lên đầu -> trọng số 2
        not_all += [a for a in se.forbidden_attrs_en if a]
    have_all, not_all = have_all[:8], not_all[:6]
    calib = calib or {}
    # thuộc tính mà chính ảnh THẬT cũng không đạt thì không kiểm được bằng VLM này -> bỏ khỏi bảng kiểm, ghi lại
    dead = [a for a in have_all if calib.get(a, {}).get("checkable") is False]
    if dead and len(dead) == len(have_all):
        dead = []      # bỏ hết thì Reviewer mù, mọi ảnh đều 0 điểm -> thà giữ bảng kiểm nhiễu còn hơn không có
    have_all = [a for a in have_all if a not in dead]
    thr_have = lambda a: calib.get(a, {}).get("thr", 0.60)
    thr_not = lambda a: calib.get(a, {}).get("thr", 0.70)
    w = lambda a: 2.0 if a in key_attrs else 1.0
    matched_have, matched_not, reasons = [], [], []
    vqa: dict[str, float] = {}
    contra: dict[str, float] = {}  # P(Yes) của câu phủ định đối chứng (chỉ khi không có mô tả sai đối ứng)
    fc: dict[str, float] = {}      # P(thuộc tính đúng) của câu trắc nghiệm hai lựa chọn
    alts = alts or {}
    name_en = next((se.name_en.split("(")[0].strip() for se in spec.entities if se.kind == "object"), "outfit")
    if have_all or not_all:
        try:
            m = agent.match_descriptors(compact_text(desc), have_all, not_all)
        except Exception as exc:  # noqa: BLE001 - v1.5.1 p001: Qwen trả JSON quá dài bị cắt -> RuntimeError làm dừng cả bước
            m = {}
            reasons.append(f"agent văn bản lỗi ({type(exc).__name__}), chỉ dùng luật")
        matched_have = [a for a in m.get("present_must_have", []) if a in have_all]
        matched_not = [a for a in m.get("present_must_not", []) if a in not_all]
        # luật cứng trên trường trang phục ghi đè agent văn bản (cả hai chiều)
        for a in have_all:
            r = garment_rules(desc, a)
            if r == "present" and a not in matched_have:
                matched_have.append(a)
            elif r == "absent" and a in matched_have:
                matched_have.remove(a); reasons.append(f"luật: mô tả trái với '{a[:40]}'")
        for a in not_all:
            r = garment_rules(desc, a)
            if r == "present" and a not in matched_not:
                matched_not.append(a); reasons.append(f"luật: mô tả có '{a[:40]}'")
            elif r == "absent" and a in matched_not:
                matched_not.remove(a)
        # v1.5.1 p001: Qwen 3B ghi collar=crossed cho ảnh cổ đứng rõ -> 5/8 ảnh bị loại nhầm. must_not do VLM đọc ra
        # phải được CLIP xác nhận trên chính ảnh (so cặp với must_have cùng bộ phận); CLIP không đồng ý -> không tính.
        for a in list(matched_not):
            h = _counterpart(a, have_all)
            if h is None:
                continue
            ok = clip_agrees_not(clip, desc.path, name_en, a, h)
            if ok is False:
                matched_not.remove(a)
                if h not in matched_have and garment_rules(desc, h) != "absent":
                    pass  # không tự thêm must_have; chỉ gỡ must_not sai
                reasons.append(f"VLM nói '{a[:30]}' nhưng CLIP nghiêng về '{h[:30]}' -> bỏ")
        # v1.7.1: câu hỏi có/không trên ẢNH cho từng thuộc tính (Exposing Blindspots 2026: câu hỏi phủ định cho must_not).
        # Mô tả tự do hay bỏ sót chi tiết (cổ áo, đai) -> VQA là ý kiến thứ hai: >= 0.75 xác nhận must_have, >= 0.85 thêm must_not, <= 0.25 bác.
        if hasattr(agent, "vqa_yes"):
            subj = name_en if any(se.kind == "object" for se in spec.entities) else "scene"
            # v1.9.5: mô tả tự do và đếm người vẫn đọc ảnh đầy đủ; riêng câu hỏi thuộc tính đọc ảnh đã cắt quanh người
            vp = person_crop(agent, desc.path) if calib else desc.path
            for a in have_all + not_all:
                q = attr_question(subj, a)
                pr = agent.vqa_yes(q, vp)
                if pr is None:
                    break
                # đối chứng phủ định cho must_have: gật cả hai chiều = không phân biệt được
                if a in have_all and alts.get(a):
                    pc = forced_choice(agent, subj, vp, a, alts[a])
                    if pc is not None:
                        fc[a] = round(pc, 3)
                        if abs(pc - pr) >= 0.30:
                            reasons.append(f"trắc nghiệm '{a[:24]}' {pc:.2f} thay câu có/không {pr:.2f}"
                                           f" (sai: {alts[a][:34]})")
                        pr = pc  # câu trắc nghiệm phân biệt tốt hơn -> dùng làm giá trị quyết định
                elif a in have_all and pr >= 0.60:
                    pn = agent.vqa_yes(attr_question_neg(subj, a), vp)
                    if pn is not None:
                        contra[a] = round(pn, 3)
                        if pn >= 0.55:
                            reasons.append(f"VQA gật cả hai chiều '{a[:26]}' ({pr:.2f}/{pn:.2f}) -> không tính")
                            pr = 0.5  # đưa về dải "không đo được"
                vqa[a] = round(pr, 3)
            for a in have_all:
                pr = vqa.get(a)
                if pr is None:
                    continue
                # v1.9: 0,75 rơi đúng giữa hai mode của phân bố (0,731 và 0,755) -> đổi ngưỡng xác nhận về 0,60 và ghi nhận
                # dải "không chắc" 0,35-0,60 để tính điểm liên tục thay vì nhảy bậc.
                t = thr_have(a)
                if pr >= t and a not in matched_have and garment_rules(desc, a) != "absent":
                    matched_have.append(a); reasons.append(f"VQA xác nhận '{a[:30]}' ({pr:.2f} >= {t:.2f})")
                elif pr < t and a in matched_have and garment_rules(desc, a) != "present":
                    matched_have.remove(a)
                    reasons.append(f"VQA dưới mốc ảnh thật '{a[:26]}' ({pr:.2f} < {t:.2f})")
                elif pr <= 0.25 and a in matched_have and garment_rules(desc, a) != "present":
                    matched_have.remove(a); reasons.append(f"VQA bác '{a[:30]}' ({pr:.2f})")
            for a in not_all:
                pr = vqa.get(a)
                if pr is None:
                    continue
                if pr >= thr_not(a) and a not in matched_not:  # v1.9.3: mốc lấy từ chính ảnh thật, sàn 0,70
                    h = _counterpart(a, have_all)
                    if h is None or clip_agrees_not(clip, desc.path, name_en, a, h) is not False:
                        matched_not.append(a); reasons.append(f"VQA thấy must_not '{a[:30]}' ({pr:.2f})")
                elif pr <= 0.25 and a in matched_not:
                    matched_not.remove(a); reasons.append(f"VQA bác must_not '{a[:30]}' ({pr:.2f})")
    missing = [a for a in have_all if a not in matched_have]
    tot = sum(w(a) for a in have_all)
    # v1.9: thuộc tính chưa khớp nhưng VQA ở dải giữa được cộng PHẦN theo xác suất -> phá hoà (S001: 16/26 ảnh cùng +1.00)
    # v1.9.3: điểm cộng PHẦN tối đa nửa trọng số. Trước đây cộng đủ trọng số nên ảnh áo liền quần (tà 0,85, ngưỡng
    # ảnh thật 0,87) vẫn được 0,97 - hiệu chỉnh ngưỡng bị điểm cộng phần vô hiệu hoá.
    partial = sum(0.5 * w(a) * max(0.0, min(1.0, (vqa[a] - (thr_have(a) - 0.25)) / 0.25))
                  for a in missing if (thr_have(a) - 0.25) < vqa.get(a, 0.0) < thr_have(a))
    score = ((sum(w(a) for a in matched_have) + partial - sum(w(a) for a in matched_not)) / tot) if tot else 0.0
    score = max(-1.0, min(1.0, score))
    keep = True
    if matched_not:
        reasons.append("có must_not: " + "; ".join(matched_not[:2]))
        if kind == "candidate" or not matched_have:
            keep = False
    if n_people == 1 and desc.people_count is not None and desc.people_count >= 3:
        reasons.append(f"prompt một người, ảnh {desc.people_count} người")
        keep = False
    if kind == "reference" and desc.watermark_or_text:
        reasons.append("có chữ/watermark")
    if dead:
        reasons.append("không kiểm được trên ảnh thật, bỏ khỏi bảng kiểm: " + "; ".join(a[:30] for a in dead))
    if not any(r.startswith(("có must_not", "prompt", "có chữ")) for r in reasons):
        reasons.append(f"{len(matched_have)}/{len(have_all)} must_have thấy trong mô tả")
    return FilterVerdict(path=desc.path, keep=keep, matched_must_have=matched_have, matched_must_not=matched_not,
                         missing_must_have=missing, people_count=desc.people_count, reasons=reasons, score=round(score, 3),
                         vqa=vqa, vqa_neg=contra, vqa_fc=fc, alt_attrs={k: alts[k] for k in fc},
                         unverifiable=dead)


#: Câu trắc nghiệm hai lựa chọn: TẮT mặc định. Đo trên 23 ảnh S001 (11 ảnh có nhãn tay): với mô tả sai viết bằng tay
#: cho đúng một đặc trưng ("one-piece outfit joined from top to bottom, no separate panels") nó tách rất tốt
#: (0,05-0,24 với ảnh sai, 0,84-0,96 với ảnh đúng). Nhưng mô tả sai sinh tự động thì hỏng: must_not ghép theo vùng
#: ("one-piece dress with no trousers underneath") lật HAI đặc trưng cùng lúc, mà ảnh áo dài thật phần lớn không
#: nhìn thấy quần -> VLM chọn nhầm, AUC 0,29 (dưới mức ngẫu nhiên). LLM viết thì lật nhầm chiều dài tay áo.
#: Giữ lại mã và bài đo; bật khi có cách sinh mô tả sai chỉ lật ĐÚNG đặc trưng phân biệt.
FORCED_CHOICE = False

#: Ngưỡng sàn/trần khi hiệu chỉnh ngưỡng theo ảnh thật, và mức dưới mà thuộc tính bị coi là không kiểm được.
_CAL_FLOOR, _CAL_CEIL, _CAL_MARGIN, _CAL_MIN_REF = 0.55, 0.90, 0.08, 0.50
#: ảnh tham chiếu đều là ví dụ ĐÚNG, nên thuộc tính đáng tin phải cho điểm GIỐNG NHAU trên cả ba.
#: Chênh lệch lớn = không quan sát được ổn định (quần: 0,96/0,00/0,20 vì tà áo che kín).
_CAL_MAX_SPREAD = 0.45
_CAL_CACHE: dict[tuple, dict] = {}


_CROP_MEMO: dict[str, str] = {}


def person_crop(agent, path: str, pad: float = 0.06) -> str:
    """Cắt quanh người rồi mới hỏi VQA (v1.9.5). Ảnh 1536px bị thu về ~700px trước khi vào VLM, người chiếm chưa tới
    một phần ba khung nên cổ áo và khe xẻ hông gần như biến mất. Đo trên 14 ảnh S001 có nhãn tay, AUC từng thuộc tính:
    tà xẻ 0,67 -> 0,82; thân áo 0,94 -> 0,97; cổ đứng 0,36 -> 0,58. Không tìm thấy người thì trả lại ảnh gốc."""
    if path in _CROP_MEMO:
        return _CROP_MEMO[path]
    _CROP_MEMO[path] = path              # đặt trước để lỗi cũng không thử lại
    if getattr(agent, "locate", None) is None:
        return path
    try:
        import hashlib
        import tempfile
        from PIL import Image

        out = Path(tempfile.gettempdir()) / "ctig_crops"
        out.mkdir(parents=True, exist_ok=True)
        dst = out / (hashlib.sha1(path.encode()).hexdigest()[:16] + ".png")
        if dst.exists():
            _CROP_MEMO[path] = str(dst)
            return str(dst)
        boxes = agent.locate(path, ["person"]) or agent.locate(path, ["woman"])
        if not boxes:
            return path
        im = Image.open(path).convert("RGB")
        W, H = im.size
        x0, y0, x1, y1 = max(boxes, key=lambda b: (b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1]))["bbox"]
        pw, ph = (x1 - x0) * pad, (y1 - y0) * pad
        box = (max(0, int(x0 - pw)), max(0, int(y0 - ph)), min(W, int(x1 + pw)), min(H, int(y1 + ph)))
        if (box[2] - box[0]) < 0.10 * W or (box[3] - box[1]) < 0.10 * H:
            return path              # hộp quá nhỏ: nhiều khả năng định vị sai
        im.crop(box).save(dst)
        _CROP_MEMO[path] = str(dst)
    except Exception:  # noqa: BLE001
        pass
    return _CROP_MEMO[path]


def calibrate(agent, spec: CulturalSpec, refs: list[str], log=print) -> dict[str, dict]:
    """Hiệu chỉnh ngưỡng từng thuộc tính trên ẢNH THẬT của chính prompt này (v1.9.3).

    Câu có/không XẾP HẠNG khá tốt nhưng CHUẨN ĐỘ thì sai: trên S001, thuộc tính "tunic split at the hips into front
    and back panels" cho 0,82-0,96 với ảnh đúng và 0,62-0,85 với ảnh sai (AUC 0,92) - thứ tự đúng, nhưng ngưỡng cố
    định 0,60 cho TẤT CẢ đi qua, nên ảnh áo liền quần được +1,00. Ảnh tham chiếu thật là mốc: ứng viên phải giống
    thuộc tính ít nhất gần bằng ảnh thật yếu nhất.

    Đồng thời phát hiện thuộc tính KHÔNG KIỂM ĐƯỢC: "worn over wide-legged long trousers" chỉ được 0,96/0,00/0,20
    trên ba ảnh áo dài thật, vì tà áo che kín quần. Chấm nó là chấm nhiễu -> loại khỏi bảng kiểm, vẫn giữ trong prompt.
    """
    if not refs or getattr(agent, "vqa_yes", None) is None:
        return {}
    attrs: list[tuple[str, str]] = []
    subj = next((se.name_en.split("(")[0].strip() for se in spec.entities if se.kind == "object"), "outfit")
    for se in spec.entities:
        attrs += [(a, "have") for a in se.required_attrs_en if a]
        attrs += [(a, "not") for a in se.forbidden_attrs_en if a]
    if not attrs:
        return {}
    refs = list(dict.fromkeys(refs))[:4]
    ck = (subj, tuple(sorted(attrs)), tuple(refs))
    if ck in _CAL_CACHE:
        return _CAL_CACHE[ck]
    out: dict[str, dict] = {}
    rc = [person_crop(agent, r) for r in refs]
    for a, side in attrs:
        vals = [v for v in (agent.vqa_yes(attr_question(subj, a), r) for r in rc) if v is not None]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        spread = max(vals) - min(vals)
        if side == "have":
            out[a] = {"side": side, "ref_mean": round(mean, 3), "ref_min": round(min(vals), 3),
                      "ref_spread": round(spread, 3),
                      "checkable": mean >= _CAL_MIN_REF and spread <= _CAL_MAX_SPREAD,
                      "thr": round(max(_CAL_FLOOR, min(_CAL_CEIL, mean - _CAL_MARGIN)), 3)}
        else:
            # must_not phải hiếm trên ảnh thật; nếu nó đã kêu sẵn ở đó thì trên ứng viên phải kêu to hơn hẳn
            out[a] = {"side": side, "ref_mean": round(mean, 3), "ref_min": round(min(vals), 3),
                      "ref_spread": round(spread, 3), "checkable": True,
                      "thr": round(max(0.70, min(0.95, mean + 0.20)), 3)}
    dead = [a for a, c in out.items() if c["side"] == "have" and not c["checkable"]]
    log(f"  [filter] hiệu chỉnh trên {len(refs)} ảnh thật: "
        + "; ".join(f"{a[:24]} ngưỡng {c['thr']:.2f} (ảnh thật {c['ref_mean']:.2f})"
                    for a, c in list(out.items())[:3]))
    if dead:
        log("  [filter] thuộc tính KHÔNG kiểm được trên ảnh thật, bỏ khỏi bảng kiểm: "
            + "; ".join(f"{a[:34]} (TB {out[a]['ref_mean']:.2f}, chênh {out[a]['ref_spread']:.2f})" for a in dead))
    _CAL_CACHE[ck] = out
    return out


def _alternatives(agent, spec: CulturalSpec, log=print) -> dict[str, str]:
    """Một lần mỗi lô: với mỗi must_have, lấy mô tả sai cụ thể để làm lựa chọn B của câu trắc nghiệm.

    Ưu tiên ghép bằng luật từ chính must_not (đã là mô tả sai đúng vùng, do người viết hoặc rút từ web);
    chỉ hỏi LLM cho thuộc tính không có must_not cùng vùng, và chỉ nhận câu LLM viết nếu nó nói về cùng vùng.
    """
    if not FORCED_CHOICE or getattr(agent, "vqa_choice", None) is None:
        return {}
    out: dict[str, str] = {}
    n_rule = 0
    for se in spec.entities:
        attrs = [a for a in se.required_attrs_en if a]
        nots = [a for a in se.forbidden_attrs_en if a]
        if not attrs:
            continue
        paired = pair_distractors(attrs, nots)
        out.update(paired)
        n_rule += len(paired)
        rest = [a for a in attrs if a not in paired]
        if not rest or getattr(agent, "attr_alternatives", None) is None:
            continue
        try:
            got = agent.attr_alternatives(se.name_en.split("(")[0].strip(), rest, nots)
        except Exception as exc:  # noqa: BLE001
            log(f"  [filter] không lấy được mô tả sai đối ứng ({type(exc).__name__})")
            continue
        for a, alt in got.items():
            if a in out or a not in rest:
                continue
            if not usable_distractor(a, alt):
                log(f"  [filter] bỏ mô tả sai lệch vùng hoặc chỉ phủ định: '{a[:28]}' vs '{alt[:32]}'")
                continue
            out[a] = alt
    if out:
        log(f"  [filter] trắc nghiệm hai lựa chọn cho {len(out)} thuộc tính ({n_rule} ghép từ must_not): "
            + "; ".join(f"{k[:22]} vs {v[:28]}" for k, v in list(out.items())[:3]))
    return out


def run(agent, paths: list[str], spec: CulturalSpec, prompt_en: str, kind: str = "candidate", log=print, clip=None,
        refs: list[str] | None = None) -> FilterResult:
    n_people = expected_people(prompt_en)
    paths = list(dict.fromkeys(paths))  # hàng alias (+ref bị gate) chia sẻ đường dẫn -> không mô tả hai lần
    descs = describe(agent, paths, log=log)
    alts = _alternatives(agent, spec, log=log)
    # ảnh thật của chính prompt này làm mốc; khi đang lọc chính ảnh thật thì không hiệu chỉnh (vòng tròn)
    calib = calibrate(agent, spec, refs or [], log=log) if kind != "reference" else {}
    verdicts = [_verdict(agent, d, spec, kind, n_people, clip=clip, alts=alts, calib=calib) for d in descs]
    kept = [v.path for v in verdicts if v.keep]
    if not kept and verdicts:
        best = max(verdicts, key=lambda v: v.score)
        best.keep = True
        best.reasons.append("giữ lại vì là ảnh tốt nhất còn lại")
        kept = [best.path]
    log(f"  [filter:{kind}] giữ {len(kept)}/{len(paths)}" + (f", prompt nói rõ {n_people} người" if n_people else ""))
    return FilterResult(kind=kind, expected_people=n_people, verdicts=verdicts, kept=kept, descriptors=descs,
                        calibration=calib)
