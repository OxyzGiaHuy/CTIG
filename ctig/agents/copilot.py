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
             threshold: float = DEFAULT_THRESHOLD, log=print) -> EvalResult:
    """Ba trục. Trục văn hoá đối chiếu với ẢNH THẬT, không với bảng must_have."""
    ev = EvalResult(path=image)
    img = crop or image
    refs = [r for r in (refs or [])][:3]

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
        ev.foreign = [x for x in (_clean_item(v, allow_absence=False) for v in (d2.get("foreign_elements") or [])) if x][:2]
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


def suggestions(ev: EvalResult) -> str:
    """Góp ý gửi về bộ sinh, đúng vai 'improvement suggestions' của T2I-Copilot."""
    bits = []
    if ev.differences:
        bits.append("Fix on the object: " + "; ".join(ev.differences) + ".")
    if ev.foreign:
        bits.append("Remove details from other cultures: " + "; ".join(ev.foreign) + ".")
    if ev.identity_p < 0.5:
        bits.append(f"The object must clearly read as the Vietnamese one, not as {ev.identity}.")
    low = [k for k, v in ev.prompt_scores.items() if v < 6]
    if low:
        bits.append("Weak on: " + ", ".join(low).replace("_", " ") + ".")
    return " ".join(bits)


# ------------------------------------------------------------------ 3. Generation Engine, vòng lặp
def run_loop(agent, report: dict, first_image: str, generate, refs=None, crop=None,
             threshold: float = DEFAULT_THRESHOLD, max_rounds: int = DEFAULT_MAX_ROUNDS, log=print) -> dict:
    """Vòng lặp đúng kiểu T2I-Copilot: chấm, dưới ngưỡng thì SINH LẠI kèm góp ý. Không có thang leo.

    `generate(suggestion_text, round_index) -> đường dẫn ảnh mới hoặc None`; `crop(path) -> path` không bắt buộc.
    """
    cr = lambda p: (crop(p) if crop else p)  # noqa: E731
    best_img, rounds = first_image, []
    ev = evaluate(agent, first_image, report, refs, cr(first_image), threshold, log)
    best = ev
    if ev.passed:
        log(f"  [loop] ảnh đầu đạt ({ev.overall:.1f} >= {threshold}) -> dừng")
        return {"final": first_image, "best": best.to_dict(), "rounds": rounds, "stop": "ảnh đầu đạt"}
    for n in range(1, max_rounds + 1):
        tip = suggestions(ev)
        log(f"  [loop] vòng {n}: {tip[:150]}")
        new = generate(tip, n)
        if not new:
            rounds.append({"n": n, "suggestion": tip, "note": "sinh lại không ra ảnh"})
            log(f"  [loop] vòng {n}: sinh lại không ra ảnh -> dừng")
            break
        ev = evaluate(agent, new, report, refs, cr(new), threshold, log)
        rounds.append({"n": n, "suggestion": tip, "image": new, "eval": ev.to_dict()})
        if ev.overall > best.overall:
            best, best_img = ev, new
        if ev.passed:
            return {"final": best_img, "best": best.to_dict(), "rounds": rounds, "stop": f"đạt ở vòng {n}"}
    return {"final": best_img, "best": best.to_dict(), "rounds": rounds,
            "stop": f"hết {max_rounds} vòng, giữ ảnh tốt nhất {best.overall:.1f}"}
