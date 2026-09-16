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
             alts: dict[str, str] | None = None) -> FilterVerdict:
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
            for a in have_all + not_all:
                q = attr_question(subj, a)
                pr = agent.vqa_yes(q, desc.path)
                if pr is None:
                    break
                # đối chứng phủ định cho must_have: gật cả hai chiều = không phân biệt được
                if a in have_all and alts.get(a):
                    pc = forced_choice(agent, subj, desc.path, a, alts[a])
                    if pc is not None:
                        fc[a] = round(pc, 3)
                        if abs(pc - pr) >= 0.30:
                            reasons.append(f"trắc nghiệm '{a[:24]}' {pc:.2f} thay câu có/không {pr:.2f}"
                                           f" (sai: {alts[a][:34]})")
                        pr = pc  # câu trắc nghiệm phân biệt tốt hơn -> dùng làm giá trị quyết định
                elif a in have_all and pr >= 0.60:
                    pn = agent.vqa_yes(attr_question_neg(subj, a), desc.path)
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
                if pr >= 0.60 and a not in matched_have and garment_rules(desc, a) != "absent":
                    matched_have.append(a); reasons.append(f"VQA xác nhận '{a[:30]}' ({pr:.2f})")
                elif pr <= 0.25 and a in matched_have and garment_rules(desc, a) != "present":
                    matched_have.remove(a); reasons.append(f"VQA bác '{a[:30]}' ({pr:.2f})")
            for a in not_all:
                pr = vqa.get(a)
                if pr is None:
                    continue
                if pr >= 0.70 and a not in matched_not:  # v1.9: 0,85 bỏ sót 25 câu trong dải 0,60-0,85 (S031); 0,70 cân bằng hơn
                    h = _counterpart(a, have_all)
                    if h is None or clip_agrees_not(clip, desc.path, name_en, a, h) is not False:
                        matched_not.append(a); reasons.append(f"VQA thấy must_not '{a[:30]}' ({pr:.2f})")
                elif pr <= 0.25 and a in matched_not:
                    matched_not.remove(a); reasons.append(f"VQA bác must_not '{a[:30]}' ({pr:.2f})")
    missing = [a for a in have_all if a not in matched_have]
    tot = sum(w(a) for a in have_all)
    # v1.9: thuộc tính chưa khớp nhưng VQA ở dải giữa được cộng PHẦN theo xác suất -> phá hoà (S001: 16/26 ảnh cùng +1.00)
    partial = sum(w(a) * max(0.0, min(1.0, (vqa[a] - 0.35) / 0.25)) for a in missing if 0.35 < vqa.get(a, 0.0) < 0.60)
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
    if not any(r.startswith(("có must_not", "prompt", "có chữ")) for r in reasons):
        reasons.append(f"{len(matched_have)}/{len(have_all)} must_have thấy trong mô tả")
    return FilterVerdict(path=desc.path, keep=keep, matched_must_have=matched_have, matched_must_not=matched_not,
                         missing_must_have=missing, people_count=desc.people_count, reasons=reasons, score=round(score, 3),
                         vqa=vqa, vqa_neg=contra, vqa_fc=fc, alt_attrs={k: alts[k] for k in fc})


def _alternatives(agent, spec: CulturalSpec, log=print) -> dict[str, str]:
    """Một lần mỗi lô: với mỗi must_have, lấy mô tả sai cụ thể để làm lựa chọn B của câu trắc nghiệm.

    Ưu tiên ghép bằng luật từ chính must_not (đã là mô tả sai đúng vùng, do người viết hoặc rút từ web);
    chỉ hỏi LLM cho thuộc tính không có must_not cùng vùng, và chỉ nhận câu LLM viết nếu nó nói về cùng vùng.
    """
    if getattr(agent, "vqa_choice", None) is None:
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


def run(agent, paths: list[str], spec: CulturalSpec, prompt_en: str, kind: str = "candidate", log=print, clip=None) -> FilterResult:
    n_people = expected_people(prompt_en)
    paths = list(dict.fromkeys(paths))  # hàng alias (+ref bị gate) chia sẻ đường dẫn -> không mô tả hai lần
    descs = describe(agent, paths, log=log)
    alts = _alternatives(agent, spec, log=log)
    verdicts = [_verdict(agent, d, spec, kind, n_people, clip=clip, alts=alts) for d in descs]
    kept = [v.path for v in verdicts if v.keep]
    if not kept and verdicts:
        best = max(verdicts, key=lambda v: v.score)
        best.keep = True
        best.reasons.append("giữ lại vì là ảnh tốt nhất còn lại")
        kept = [best.path]
    log(f"  [filter:{kind}] giữ {len(kept)}/{len(paths)}" + (f", prompt nói rõ {n_people} người" if n_people else ""))
    return FilterResult(kind=kind, expected_people=n_people, verdicts=verdicts, kept=kept, descriptors=descs)
