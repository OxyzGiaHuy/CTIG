"""Agentic Loop v2 — ba agent theo khung T2I-Copilot (arXiv 2507.20536), KHÔNG dùng KB viết tay.

Khác hẳn vòng lặp v1.9.x: ở đó đích sửa là `must_have_en` / `must_not_en` trong `data/kb/entities.json`, nên
phải hiệu chỉnh ngưỡng cho từng thuộc tính và hệ thống tự chấm chính cái nó tối ưu. Ở đây không có bảng kiểm
nào cả. Chuẩn đối chiếu về văn hoá là **ẢNH THẬT của chính prompt** và tư liệu đã truy hồi.

Ba agent, đặt theo T2I-Copilot (tên hàm dễ đổi nếu muốn gọi khác):

  interpret()  ~ Input Interpreter   prompt + tư liệu -> Analysis Report JSON, kèm danh sách vật DỄ NHẦM
  evaluate()   ~ Quality Evaluator   chấm ảnh trên BA trục, trả góp ý cụ thể
  run_loop()   ~ Generation Engine   dưới ngưỡng thì sinh lại kèm góp ý, tối đa `max_rounds` vòng

Trục thứ ba (văn hoá) là phần ta thêm vào; 10 tiểu mục của T2I-Copilot đều trung tính về văn hoá nên sẽ chấm
đạt cho một chiếc qipao khi prompt nói áo dài.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: T2I-Copilot chấm 0-10 mỗi tiểu mục rồi so trung bình với ngưỡng 8,0, tối đa 3 vòng.
DEFAULT_THRESHOLD = 8.0
DEFAULT_MAX_ROUNDS = 3
#: Trục văn hoá nhân đôi: đó là thứ bài này quan tâm, hai trục kia là của họ.
AXIS_WEIGHT = {"prompt": 1.0, "aesthetic": 1.0, "culture": 2.0}

PROMPT_FIELDS = ("main_subjects_present", "spatial_relationships", "style_adherence", "background")
AESTHETIC_FIELDS = ("composition", "color_harmony", "lighting", "sharpness")


@dataclass
class EvalResult:
    path: str
    prompt_scores: dict[str, float] = field(default_factory=dict)
    aesthetic_scores: dict[str, float] = field(default_factory=dict)
    identity: str = ""                 # nhãn model chọn khi bị ép chọn giữa các nền văn hoá
    identity_p: float = 0.0            # xác suất của lựa chọn ĐÚNG (thực thể Việt)
    differences: list[str] = field(default_factory=list)   # khác ảnh thật ở đâu, trên chính vật thể
    foreign: list[str] = field(default_factory=list)       # chi tiết thuộc văn hoá khác
    axes: dict[str, float] = field(default_factory=dict)
    overall: float = 0.0
    passed: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)


#: Vật ở CẠNH thực thể, không phải cấu tạo của nó. Lượt thử S012 trả về "no net", "no fish", "no hat" —
#: đó là đồ vật trong ảnh tham chiếu, không nói gì về việc chiếc thuyền thúng được đóng đúng hay sai.
_NEARBY = ("net", "fish", "hat", "person", "people", "man", "woman", "fisherman", "clothing", "clothes",
           "cargo", "basket of", "tool", "oar in", "water", "sky", "background", "lighting", "shadow")


def _clean_item(v, allow_absence: bool = True) -> str:
    t = " ".join(str(v or "").split()).strip(" .;")
    if not t or len(t.split()) < 2:
        return ""
    low = t.lower()
    if not allow_absence and low.startswith(("no ", "not ", "missing", "lack", "absence", "without")):
        return ""                        # ô "văn hoá khác" chỉ nhận thứ NHÌN THẤY, không nhận thứ thiếu
    if any(w in low for w in _NEARBY) and not any(w in low for w in ("hull", "shape", "weave", "material",
                                                                    "frame", "rim", "bottom", "side")):
        return ""                        # nói về đồ vật cạnh bên, không nói về cấu tạo thực thể
    return t


#: Ô "chi tiết thuộc văn hoá khác" chỉ có nghĩa nếu nói ĐƯỢC đó là văn hoá nào. Lượt S001 đầu tiên trả về
#: "bó hoa đỏ" và "quyển sách đen" — đồ vật trong cảnh, không phải dấu hiệu của nền văn hoá nào, mà vẫn kéo
#: trục văn hoá từ 10 xuống 0. Bắt buộc nêu tên thì lọc được bằng máy, và cũng đúng với chữ "DIFFERENT NAMED
#: culture" trong câu lệnh gửi cho model.
_CULTURES = ("chinese", "china", "japanese", "japan", "korean", "korea", "thai", "thailand", "indian", "india",
             "western", "european", "american", "arab", "persian", "turkish", "mongolian", "tibetan", "khmer",
             "cambodian", "lao", "laotian", "burmese", "myanmar", "malay", "indonesian", "filipino", "russian",
             "french", "british", "qipao", "cheongsam", "hanbok", "kimono", "yukata", "sari", "obi", "hanfu",
             "geisha", "samurai", "mandarin square", "dragon motif", "cherry blossom")


def _names_a_culture(text: str) -> bool:
    low = text.lower()
    return any(c in low for c in _CULTURES)


def _tag(path: str) -> str:
    """Nhãn ngắn cho log: tên thư mục vòng nếu có, không thì tên tệp."""
    from pathlib import Path as _P

    q = _P(path)
    for part in reversed(q.parts[:-1]):
        if part.startswith("iter"):
            return part
    return q.stem


def _num(x, lo=0.0, hi=10.0, default=5.0) -> float:
    try:
        return max(lo, min(hi, float(x)))
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------------ 1. Input Interpreter
def interpret(agent, prompt_vi: str, prompt_en: str, refined_en: str = "", evidence: str = "", log=print) -> dict:
    """Analysis Report như T2I-Copilot, thêm `look_alikes` suy TỪ TƯ LIỆU chứ không tra KB."""
    system = (
        "You prepare a structured analysis report for a text-to-image system. Read the prompt and the reference "
        "notes, then list what the picture must show. Be concrete and visual. For the main cultural object, also "
        "list the objects from OTHER cultures that image models most often draw by mistake instead of it. "
        "Each look-alike must be a NAMED object of a NAMED culture, for example 'a Japanese kimono', "
        "'a Chinese qipao', 'a Welsh coracle', 'an Indian parisal'. Do not write generic categories such as "
        "'a wooden boat' or 'a long dress' - a generic answer makes the check useless."
    )
    user = (f"PROMPT (Vietnamese): {prompt_vi}\nPROMPT (English): {prompt_en}\n"
            + (f"EXPANDED PROMPT: {refined_en}\n" if refined_en else "")
            + (f"REFERENCE NOTES: {evidence[:1500]}\n" if evidence else "")
            + 'Return JSON: {"subjects": [..], "attributes": [..], "spatial": [..], "background": "..", '
              '"style": "..", "cultural_entity_en": "..", "look_alikes": [".." up to 4]}')
    schema = {"type": "object", "properties": {
        "subjects": {"type": "array", "items": {"type": "string"}},
        "attributes": {"type": "array", "items": {"type": "string"}},
        "spatial": {"type": "array", "items": {"type": "string"}},
        "background": {"type": "string"}, "style": {"type": "string"},
        "cultural_entity_en": {"type": "string"},
        "look_alikes": {"type": "array", "items": {"type": "string"}}}, "required": ["cultural_entity_en"]}
    try:
        d = agent._complete(system, user, schema, max_new_tokens=600) or {}
    except Exception as exc:  # noqa: BLE001
        log(f"  [interpreter] lỗi {type(exc).__name__} -> báo cáo rỗng")
        d = {}
    rep = {
        "subjects": [str(x) for x in (d.get("subjects") or [])][:6],
        "attributes": [str(x) for x in (d.get("attributes") or [])][:8],
        "spatial": [str(x) for x in (d.get("spatial") or [])][:4],
        "background": str(d.get("background") or ""),
        "style": str(d.get("style") or ""),
        "entity_en": str(d.get("cultural_entity_en") or "").strip(),
        "look_alikes": [str(x).strip() for x in (d.get("look_alikes") or []) if str(x).strip()][:4],
    }
    log(f"  [interpreter] thực thể '{rep['entity_en'][:34]}' · dễ nhầm với: "
        + (", ".join(rep["look_alikes"]) or "(không có)"))
    return rep


# ------------------------------------------------------------------ 2. Quality Evaluator
def _identity_question(entity_en: str, look_alikes: list[str]) -> tuple[str, list[str]]:
    opts = [entity_en] + list(look_alikes[:4]) + ["none of these / unclear"]
    letters = "ABCDEF"[: len(opts)]
    lines = "\n".join(f"{L}. {o}" for L, o in zip(letters, opts))
    q = ("Look at the main object in this photo. Which ONE does it most resemble?\n" + lines
         + f"\nAnswer with only the letter {letters[0]} to {letters[-1]}.")
    return q, opts


def evaluate(agent, image: str, report: dict, refs: list[str] | None = None, crop: str | None = None,
             threshold: float = DEFAULT_THRESHOLD, log=print, n_refs: int = 2) -> EvalResult:
    """Ba trục. Trục văn hoá đối chiếu với ẢNH THẬT, không với bảng must_have.

    `refs` phải là ảnh thật ĐÃ CẮT quanh chủ thể, giống ảnh sinh. Lượt chạy S012 so ảnh sinh đã cắt với ảnh
    thật nguyên khung, và Qwen2.5-VL-7B phải nhìn 4 ảnh một lúc ở độ phân giải thấp: nó lặp lại "hull is oval"
    cho đúng tấm ảnh thuyền tròn rõ ràng. Cắt cả hai phía và giảm còn 2 ảnh thật để mỗi ảnh được nhiều pixel hơn.
    """
    ev = EvalResult(path=image)
    img = crop or image
    refs = [r for r in (refs or [])][:n_refs]

    # --- trục 1 + 2: khớp prompt và thẩm mỹ, đúng 8 tiểu mục kiểu T2I-Copilot
    want = ("; ".join(report.get("subjects") or []) + " | " + "; ".join(report.get("attributes") or [])
            + " | nền: " + (report.get("background") or ""))
    system = ("You score a generated image. Give each field an integer 0-10. Be strict: 10 only if flawless. "
              "Judge only what is visible.")
    user = ("WHAT THE PICTURE SHOULD SHOW: " + want[:900] + "\nReturn JSON with exactly these keys: "
            + ", ".join(PROMPT_FIELDS + AESTHETIC_FIELDS))
    schema = {"type": "object", "properties": {k: {"type": "number"} for k in PROMPT_FIELDS + AESTHETIC_FIELDS}}
    try:
        d = agent.llm.complete_json(system, user, schema, images=[img]) or {}
    except Exception as exc:  # noqa: BLE001
        ev.notes.append(f"chấm prompt/thẩm mỹ lỗi ({type(exc).__name__})")
        d = {}
    ev.prompt_scores = {k: _num(d.get(k)) for k in PROMPT_FIELDS}
    ev.aesthetic_scores = {k: _num(d.get(k)) for k in AESTHETIC_FIELDS}

    # --- trục 3a: ép chọn giữa thực thể đúng và các vật dễ nhầm
    entity = report.get("entity_en") or "the main object"
    q, opts = _identity_question(entity, report.get("look_alikes") or [])
    probs = None
    fn = getattr(agent, "vqa_choice", None)
    if fn is not None:
        probs = fn(q, img, len(opts))
    if probs:
        ev.identity_p = float(probs[0])
        ev.identity = opts[max(range(len(probs)), key=lambda i: probs[i])]
    else:
        ev.notes.append("không chấm được câu ép chọn thực thể")
        ev.identity_p, ev.identity = 0.5, entity

    # --- trục 3b: khác ẢNH THẬT ở đâu; đây chính là góp ý gửi về bộ sinh
    if refs:
        system2 = (
            "The FIRST image is generated. The other images are real photographs of the same cultural object.\n"
            "Report only how the OBJECT ITSELF is built differently: its shape, proportions, material, weave, "
            "structure, how its parts join. Name the part and say what is wrong with it, for example "
            "'hull is oval instead of circular' or 'sides are planked wood instead of woven bamboo'.\n"
            "NEVER mention anything that is merely near the object or carried in it: people, clothing, hats, "
            "nets, fish, cargo, tools, water, sky, background. NEVER mention lighting, pose, camera angle or "
            "image quality. NEVER report something as missing just because it appears in a photograph.\n"
            "If the object is built correctly, return an empty list.\n"
            "Separately, list details that are PRESENT in the generated image and belong to a DIFFERENT named "
            "culture. Only things you can see; never write a missing thing there.")
        user2 = (f"OBJECT: {entity}\nReturn JSON: "
                 '{"differences": [".." up to 3], "foreign_elements": [".." up to 2]}')
        schema2 = {"type": "object", "properties": {
            "differences": {"type": "array", "items": {"type": "string"}},
            "foreign_elements": {"type": "array", "items": {"type": "string"}}}}
        try:
            d2 = agent.llm.complete_json(system2, user2, schema2, images=[img] + refs) or {}
        except Exception as exc:  # noqa: BLE001
            ev.notes.append(f"so với ảnh thật lỗi ({type(exc).__name__})")
            d2 = {}
        ev.differences = [x for x in (_clean_item(v) for v in (d2.get("differences") or [])) if x][:3]
        # "no hat" từng lọt vào ô văn hoá khác: thiếu một thứ KHÔNG phải là chi tiết của nền văn hoá khác.
        ev.foreign = [x for x in (_clean_item(v, allow_absence=False) for v in (d2.get("foreign_elements") or []))
                      if x and _names_a_culture(x)][:2]
    else:
        ev.notes.append("không có ảnh thật -> bỏ tiểu mục so sánh")

    # --- gộp điểm
    ev.axes["prompt"] = sum(ev.prompt_scores.values()) / max(1, len(ev.prompt_scores))
    ev.axes["aesthetic"] = sum(ev.aesthetic_scores.values()) / max(1, len(ev.aesthetic_scores))
    culture = [10.0 * ev.identity_p]
    if refs:
        culture.append(max(0.0, 10.0 - 3.0 * len(ev.differences)))
        culture.append(10.0 if not ev.foreign else 0.0)
    ev.axes["culture"] = sum(culture) / len(culture)
    w = sum(AXIS_WEIGHT[k] for k in ev.axes)
    ev.overall = sum(AXIS_WEIGHT[k] * v for k, v in ev.axes.items()) / w
    # Đạt = trên ngưỡng VÀ không còn khiếm khuyết nêu tên được. Bộ chấm đã chỉ ra "thiếu mái chèo" thì đó là
    # việc sửa được, không có lý do dừng. T2I-Copilot chỉ so trung bình với ngưỡng, nhưng họ không có trục nào
    # trả về danh sách lỗi cụ thể; ta có, nên dùng.
    ev.passed = ev.overall >= threshold and not ev.foreign and not ev.differences
    log(f"  [evaluator] {ev.overall:.1f}/10 (prompt {ev.axes['prompt']:.1f} · thẩm mỹ {ev.axes['aesthetic']:.1f} "
        f"· văn hoá {ev.axes['culture']:.1f}) · nhận là '{ev.identity[:26]}' p={ev.identity_p:.2f}"
        + (f" · khác ảnh thật: {'; '.join(ev.differences)}" if ev.differences else "")
        + (f" · LAI: {'; '.join(ev.foreign)}" if ev.foreign else ""))
    return ev


def critique(ev: EvalResult, entity_en: str = "") -> str:
    """Bản nhận xét dạng chữ, để đọc và để đưa cho LLM viết lại. KHÔNG được đưa thẳng vào prompt sinh ảnh."""
    bits = []
    if ev.differences:
        bits.append("Wrong on the object: " + "; ".join(ev.differences) + ".")
    if ev.foreign:
        bits.append("Details from another culture present: " + "; ".join(ev.foreign) + ".")
    if ev.identity_p < 0.5:
        bits.append(f"The object currently reads as {ev.identity}, not as {entity_en or 'the Vietnamese one'}.")
    return " ".join(bits)


def suggestions(agent, ev: EvalResult, entity_en: str = "", log=print) -> tuple[str, list[str]]:
    """Biến nhận xét thành (câu MÔ TẢ ĐÚNG để cộng vào prompt, danh sách từ cho negative prompt).

    Đây là chỗ đã sai ở lượt chạy S012 đầu tiên: chuỗi nhận xét được ghép THẲNG vào prompt, nên bộ sinh nhận
    nguyên văn "hull is oval", "planked wood", "blue boat", "wooden fishing boat". Mô hình khuếch tán không
    có phủ định, mọi từ trong prompt đều là từ NÊN VẼ, nên ta đang yêu cầu đúng cái mình muốn bỏ. Thân thuyền
    không tròn lên được là vì vậy, không phải vì model bất lực.

    T2I-Copilot cũng không dán nhận xét vào prompt: `A_gen` của họ "refines inputs, optimizes prompts".
    """
    if not (ev.differences or ev.foreign or ev.identity_p < 0.5):
        return "", []
    txt = critique(ev, entity_en)
    system = (
        "You turn a critique of a generated image into text for a text-to-image model. Diffusion models have "
        "no negation: every word in the prompt is something to draw.\n"
        "Return two things. POSITIVE: one short phrase (max 30 words) describing how the object SHOULD look, "
        "using only the correct shape, material and construction. Never write what is wrong, never write "
        "'not', 'instead of', 'without', and never repeat the wrong words themselves.\n"
        "NEGATIVE: 2-6 short noun phrases naming exactly the wrong things to keep out.")
    user = (f"OBJECT: {entity_en or 'the main object'}\nCRITIQUE: {txt}\n"
            'Return JSON: {"positive": "..", "negative": [".."]}')
    schema = {"type": "object", "properties": {"positive": {"type": "string"},
                                               "negative": {"type": "array", "items": {"type": "string"}}}}
    pos, neg = "", []
    try:
        d = agent._complete(system, user, schema, max_new_tokens=300) or {}
        pos = " ".join(str(d.get("positive") or "").split())
        neg = [" ".join(str(x).split()) for x in (d.get("negative") or []) if str(x).strip()][:6]
    except Exception as exc:  # noqa: BLE001
        log(f"  [suggestions] LLM lỗi {type(exc).__name__}")
    bad = ("not ", "instead", "without", "no ", "avoid", "remove")
    if pos and any(b in pos.lower() for b in bad):
        log(f"  [suggestions] câu mô tả còn phủ định -> bỏ: {pos[:70]}")
        pos = ""
    neg = _drop_contradictions(pos, neg, log)
    if not pos:                           # lùi an toàn: thà không thêm gì còn hơn thêm từ sai
        neg = neg or _drop_contradictions("", [x for x in ev.differences + ev.foreign], log)
    return pos, neg


def _drop_contradictions(pos: str, neg: list[str], log=print) -> list[str]:
    """Bỏ khỏi negative những cụm mà positive đang YÊU CẦU. Hai lệnh ngược nhau thì triệt tiêu nhau.

    Gặp thật ở lượt S001 đầu tiên: positive 'V-neck collar, long sleeves, fitted skirt, silk fabric' đi kèm
    negative ['wide collar', 'long sleeves', 'wide skirt', 'black book'] — 'long sleeves' vừa được bảo vẽ vừa
    bị cấm vẽ. LLM viết hai danh sách trong một lượt nên không tự thấy mâu thuẫn.

    Luật: bỏ cụm negative nếu MỌI từ có nghĩa của nó đều đã có trong positive. 'long sleeves' bị bỏ vì cả
    'long' lẫn 'sleeves' đều nằm trong positive; 'wide collar' được giữ vì 'wide' không nằm trong đó.
    """
    if not pos or not neg:
        return neg
    filler = {"a", "an", "the", "of", "with", "and", "in", "on", "is", "are", "too"}
    pos_words = {w.strip(".,;:'\"") for w in pos.lower().split()} - filler
    out = []
    for phrase in neg:
        words = {w.strip(".,;:'\"") for w in phrase.lower().split()} - filler
        if words and words <= pos_words:
            log(f"  [suggestions] '{phrase}' vừa ở prompt dương vừa ở prompt âm -> bỏ khỏi negative")
            continue
        out.append(phrase)
    return out


# ------------------------------------------------------------------ 2b. ĐÃ THỬ VÀ BỎ: chọn bằng so cặp
# Giữ lại mã và kết quả đo để không thử lại. Trên S012, đấu vòng tròn 4 ảnh (mỗi cặp kèm một ảnh thật, hỏi
# cái nào giống hơn, hai thứ tự lấy trung bình) cho iter0=1,64 iter1=1,63 iter2=1,39 iter3=1,34 — gần như
# phẳng trên thang 0-3, và chọn đúng ảnh TỆ NHẤT (ảnh mốc). Qwen2.5-VL-7B không so được nhiều ảnh.
# Chỉ bật lại khi bộ chấm chạy bằng VLM mạnh hơn.
def compare_pair(agent, img_a: str, img_b: str, ref: str, entity_en: str = "") -> float:
    """P(ảnh A giống ảnh thật hơn ảnh B). Hỏi CẢ HAI thứ tự rồi lấy trung bình để khử thiên lệch vị trí.

    Vì sao so cặp thay vì so điểm tuyệt đối: bộ chấm cho điểm gần như đứng yên (trục khớp prompt 5,8 ở mọi
    vòng) và có lúc nói ngược chiều so sánh. VLM trả lời "cái nào giống hơn" đáng tin hơn nhiều so với "cái
    này mấy điểm" — cùng lý do mà bài T2I-Copilot phải đặt ngưỡng tay cho điểm tuyệt đối.
    """
    fn = getattr(agent.llm, "choice_prob", None)
    if fn is None:
        return 0.5
    what = entity_en or "the main object"
    q = (f"Image A and image B are generated. Image C is a real photograph of a {what}. "
         f"Which generated image shows a {what} built more like the one in the real photograph? "
         "Judge the object's shape, proportions, material and construction only. Ignore lighting, pose, "
         "camera angle and background.\nA. image A\nB. image B\nAnswer with only the letter A or B.")
    try:
        p1 = fn(q, [img_a, img_b, ref], ("A", "B"))
        p2 = fn(q, [img_b, img_a, ref], ("A", "B"))
    except Exception:  # noqa: BLE001
        return 0.5
    return (p1[0] + p2[1]) / 2.0


def pick_best(agent, images: list[str], ref: str | None, entity_en: str = "", crop=None, log=print) -> tuple[str, dict]:
    """Đấu vòng tròn từng cặp, ảnh nào thắng nhiều nhất thì chọn. Trả (ảnh, bảng điểm thắng)."""
    imgs = [i for i in dict.fromkeys(images) if i]
    if len(imgs) < 2 or not ref:
        return (imgs[0] if imgs else ""), {}
    cr = (lambda p: crop(p)) if crop else (lambda p: p)
    cim = {i: cr(i) for i in imgs}
    cref = cr(ref)
    wins = {i: 0.0 for i in imgs}
    for a in range(len(imgs)):
        for b in range(a + 1, len(imgs)):
            ia, ib = imgs[a], imgs[b]
            p = compare_pair(agent, cim[ia], cim[ib], cref, entity_en)
            wins[ia] += p
            wins[ib] += 1.0 - p
    best = max(imgs, key=lambda i: wins[i])
    log("  [chọn cuối] đấu cặp: " + " · ".join(f"{_tag(i)}={wins[i]:.2f}" for i in imgs))
    return best, {i: round(wins[i], 3) for i in imgs}


# ------------------------------------------------------------------ 3. Generation Engine, vòng lặp
def run_loop(agent, report: dict, first_image: str, generate, refs=None, crop=None,
             threshold: float = DEFAULT_THRESHOLD, max_rounds: int = DEFAULT_MAX_ROUNDS, log=print) -> dict:
    """Vòng lặp đúng kiểu T2I-Copilot: chấm, dưới ngưỡng thì SINH LẠI kèm góp ý. Không có thang leo.

    `generate(positive_text, negative_terms, round_index) -> đường dẫn ảnh mới hoặc None`;
    `crop(path) -> path` không bắt buộc.
    """
    cr = lambda p: (crop(p) if crop else p)  # noqa: E731
    entity = report.get("entity_en") or ""
    refs = [cr(r) for r in (refs or [])]      # ảnh thật cũng phải cắt, để so cùng khung với ảnh sinh
    best_img, rounds, kept = first_image, [], [first_image]
    stop = "chưa chạy vòng nào"
    ev = evaluate(agent, first_image, report, refs, cr(first_image), threshold, log)
    best = ev
    if ev.passed:
        log(f"  [loop] ảnh đầu đạt ({ev.overall:.1f} >= {threshold}) -> dừng")
        return {"final": first_image, "best": best.to_dict(), "rounds": rounds, "kept": kept,
                "stop": "ảnh đầu đạt"}
    for n in range(1, max_rounds + 1):
        pos, neg = suggestions(agent, ev, entity, log)
        tip = critique(ev, entity)
        log(f"  [loop] vòng {n}: nhận xét: {tip[:110]}")
        log(f"  [loop] vòng {n}: thêm vào prompt: '{pos[:90]}' · negative: {neg}")
        new = generate(pos, neg, n)
        if not new:
            rounds.append({"n": n, "critique": tip, "positive": pos, "negative": neg,
                           "note": "sinh lại không ra ảnh"})
            log(f"  [loop] vòng {n}: sinh lại không ra ảnh -> dừng")
            break
        ev = evaluate(agent, new, report, refs, cr(new), threshold, log)
        rounds.append({"n": n, "critique": tip, "positive": pos, "negative": neg,
                       "image": new, "eval": ev.to_dict()})
        kept.append(new)
        if ev.overall > best.overall:
            best, best_img = ev, new
        if ev.passed:
            stop = f"đạt ở vòng {n}"
            break
    else:
        stop = f"hết {max_rounds} vòng"
    # Ảnh cuối = ảnh ĐIỂM CAO NHẤT trong mọi vòng, không phải ảnh vòng cuối. Vòng 3 sinh kém thì vẫn giữ
    # ảnh vòng 1 hoặc 2. Đã thử chọn bằng đấu cặp (mỗi cặp kèm một ảnh thật, hỏi cái nào giống hơn): điểm ra
    # gần như phẳng 1,34-1,64 trên thang 0-3 và nó chọn đúng ảnh TỆ NHẤT, nên bỏ.
    log(f"  [chọn cuối] {_tag(best_img)} điểm {best.overall:.1f} trong {len(kept)} ảnh: "
        + " · ".join(f"{_tag(i)}" for i in kept))
    return {"final": best_img, "best": best.to_dict(), "rounds": rounds, "kept": kept,
            "stop": stop + f", giữ ảnh tốt nhất {_tag(best_img)} ({best.overall:.1f})"}
