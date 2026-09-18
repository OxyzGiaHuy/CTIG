"""VietRepair — sửa ảnh MỘT LƯỢT bằng ba agent phân vai, giao tiếp qua JSON có cấu trúc.

Thay cho vòng lặp ba lượt của `copilot.py`. Lý do đổi, dựa trên số đo ngày 2026-09-17:

- Vòng lặp ba lượt sinh lại toàn ảnh mỗi vòng rồi lấy argmax, nên `điểm_C >= điểm_B` là ĐỒNG NHẤT THỨC
  toán học, không phải kết quả. Ở đây chỉ sinh lại ĐÚNG MỘT LẦN nên hết chuyện đó.
- Lấy max của 4 ảnh cho nhánh C trong khi A/B chỉ 1 ảnh cho +0,25 điểm miễn phí, lớn hơn mọi biên thắng
  thua quan sát được. Ở đây mọi nhánh đúng một ảnh, cùng seed.
- Vòng sửa làm ảnh TỆ ĐI ở 2/3 prompt. Ở đây có đường lùi no-op: không tìm được lỗi sửa được thì trả về
  đúng prompt gốc với đúng seed, nên nhánh M không bao giờ tệ hơn nhánh nền.
- Bộ chấm bịa chuẩn văn hoá ("xẻ tà không phổ biến ở áo dài", "đòn gánh khác yoke"). Ở đây Critic CHỈ được
  viện dẫn `contract_id` có thật trong contract; mọi mục bịa ra đều bị loại bằng máy.
- 99/100 prompt Culture-TRIP vượt 77 token CLIP. Ở đây prompt gốc BẤT BIẾN, và mệnh đề sửa đặt lên TRƯỚC
  nó, vì compel cắt embedding gộp ở 77 token nên phần đuôi gần như không tác dụng (S002 tả rõ 'a balanced
  pole across her shoulders' mà vẫn ra xe đạp). Chỉ mệnh đề bị giới hạn 25 từ, phần gốc không đụng tới.

Ba agent dùng CHUNG một MLLM, chỉ khác system prompt. Phải khai đúng như vậy trong bài:
"role-specialized agents sharing the same MLLM backbone", đừng gọi là ba model độc lập.

    I0 ──► A1 Observer ──M1──► A2 Critic ──M2──► A3 Refiner ──M3──► A2 duyệt ──► I1
           chỉ tả pixel        soi contract       viết mệnh đề        một lần

Bốn nhánh để chứng minh sơ đồ không phải trang trí (xem `scripts/run_arms.py`):
    B  không agent          ·  T  Refiner có contract nhưng KHÔNG nhìn ảnh
    S  một VLM tự viết      ·  M  đủ Observer -> Critic -> Refiner -> duyệt
M > B nói phản hồi có ích; M > T nói NHÌN ẢNH có ích; M > S nói PHÂN VAI có ích.
Thiếu S thì sơ đồ ba agent chỉ là trang trí.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

#: Độ dài tối đa của MỆNH ĐỀ SỬA (từ). Prompt gốc không bị giới hạn và không bao giờ bị cắt.
MAX_TOKENS = 25
#: Critic chỉ được nêu tối đa chừng này lỗi. Nhiều hơn thì mệnh đề sửa dài và loãng.
MAX_VIOLATIONS = 2

ROOT = Path(__file__).resolve().parent.parent.parent


def load_contracts(path: str | Path | None = None) -> dict:
    p = Path(path or ROOT / "data" / "contracts.json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# ------------------------------------------------------------------ thông điệp giữa các agent
@dataclass
class Trace:
    """Toàn bộ hội thoại, ghi ra JSON để đưa vào supplementary của bài."""
    prompt_id: str
    arm: str
    m1: dict | None = None          # Observer -> Critic
    m2: dict | None = None          # Critic -> Refiner
    m3: dict | None = None          # Refiner -> Critic
    review: dict | None = None      # Critic duyệt
    repair_clause: str = ""
    negative_terms: list[str] = field(default_factory=list)
    noop: bool = True
    ly_do_noop: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)


def _json(agent, system: str, user: str, schema: dict, images=None, max_new_tokens=700) -> dict:
    """Gọi MLLM và ép ra JSON. Lỗi parse -> {} để người gọi rơi về no-op, không đoán bừa."""
    try:
        fn = agent.llm.complete_json if images else agent._complete
        d = (fn(system, user, schema, images=images, max_new_tokens=max_new_tokens) if images
             else fn(system, user, schema, max_new_tokens=max_new_tokens))
        return d or {}
    except Exception:  # noqa: BLE001
        return {}



#: Từ đồng nghĩa hay gặp ở trang phục và vật dụng. Cần vì Observer viết "pants" còn contract viết
#: "trousers", nên so chữ thuần sẽ không thấy hai cái là một.
_DONG_NGHIA = {
    "pants": "trousers", "slacks": "trousers", "trouser": "trousers",
    "tunic": "top", "blouse": "top", "shirt": "top", "gown": "dress", "robe": "dress",
    "hat": "hat", "conical": "conical", "basket": "basket", "baskets": "basket",
    "pole": "pole", "yoke": "pole", "boat": "boat", "hull": "hull",
    "sleeve": "sleeves", "slit": "slits", "panel": "panels", "colour": "color", "colours": "color",
}


def _chuan(w: str) -> str:
    w = w.strip(".,;:'\"()-").lower()
    return _DONG_NGHIA.get(w, w)


def _tu(text: str) -> set[str]:
    filler = {"a", "an", "the", "of", "with", "and", "in", "on", "is", "are", "to", "at", "for",
              "that", "this", "it", "its", "no", "not", "be", "as", "by", "from", "over", "under"}
    return {_chuan(w) for w in str(text or "").split()} - filler - {""}


def _co_can_cu(evidence: str, m1: dict) -> bool:
    """Bằng chứng của Critic phải THẬT SỰ nằm trong quan sát của Observer.

    Lỗi thật ở S001: Observer nói 'white pants' (tức CÓ quần), Critic vẫn báo vi phạm
    'worn_over_trousers' với bằng chứng 'sleek and form-fitting' — cụm đó không có trong quan sát.
    Nhắc trong system prompt là không đủ; phải kiểm bằng máy.

    Luật: bằng chứng phải chung ít nhất hai từ có nghĩa với MỘT mục quan sát nào đó.
    """
    e = _tu(evidence)
    if not e:
        return False
    for f in (m1.get("visible_features") or []) + (m1.get("uncertain_features") or []) + [m1.get("subject", "")]:
        if len(e & _tu(f)) >= 2 or e <= _tu(f):
            return True
    return False


def _bo_negative_pha_prompt(neg: list[str], base_prompt: str, contract: dict, log=print) -> list[str]:
    """Bỏ từ cấm mà thật ra ta ĐANG MUỐN vẽ.

    Lỗi thật ở S001: negative của nhánh M chứa 'white pants'. Nhưng áo dài trắng thì MẶC VỚI QUẦN
    TRẮNG, và chính contract có mục 'worn over separate long wide-legged trousers'. Cấm đúng thứ mình
    yêu cầu nên SDXL vẽ ra áo choàng không quần — ảnh tệ nhất trong cả năm nhánh.

    Luật: bỏ cụm negative nếu MỌI từ có nghĩa của nó đều nằm trong prompt gốc hoặc trong các mục
    required. Khi đó nó không cấm được gì mới, chỉ cấm mất thứ đang cần.
    'sleek and form-fitting trousers' vẫn được giữ vì 'sleek' không nằm trong hai nguồn đó.
    """
    duoc_bao_ve = _tu(base_prompt) | {w for r in contract.get("required", []) for w in _tu(r["description"])}
    out = []
    for phrase in neg:
        w = _tu(phrase)
        if w and w <= duoc_bao_ve:
            log(f"  [negative] bỏ '{phrase}' — mọi từ của nó đều là thứ prompt/contract ĐANG YÊU CẦU")
            continue
        out.append(phrase)
    return out



#: Danh từ chỉ BỘ PHẬN nhìn thấy được. Dùng để rút ra "những chỗ cần mô tả" từ contract mà KHÔNG lộ đáp án:
#: nói "hãy tả phần quần" là hướng sự chú ý, còn nói "quần phải rộng ống" mới là mớm đáp án.
_BO_PHAN = {
    "collar", "neckline", "sleeves", "sleeve", "trousers", "pants", "panels", "panel", "slits", "slit",
    "hem", "bodice", "sash", "belt", "buttons", "fabric", "silk", "pattern", "patterns", "embroidery",
    "hull", "sides", "rim", "bamboo", "weave", "bow", "stern", "paddle",
    "pole", "baskets", "basket", "trays", "load", "shoulder",
    "wheels", "seat", "pedals", "canopy", "noodles", "broth", "beef", "bowl", "herbs",
    "leaves", "string", "ties", "corners", "roof", "pillar", "pond", "stairway",
    "lanterns", "ribs", "walls", "hats", "hat", "brim", "headscarf", "scarf", "grid", "colours",
    "gongs", "drum", "strings", "puppets", "water", "stage", "boats", "poles",
}


def _cho_can_ta(contract: dict, toi_da: int = 6) -> list[str]:
    """Rút danh sách BỘ PHẬN cần mô tả từ contract, đã bóc hết đáp án.

    Vì sao cần: ở S001, Observer trả về danh sách cụm rời rạc ('white pants', 'white top with red
    patterns'). Contract hỏi "áo có mặc TRÊN quần riêng không" — một QUAN HỆ giữa hai vật, mà danh sách
    rời rạc không nói được. Critic phải suy diễn và suy sai: nó báo thiếu quần trong khi Observer đã ghi
    rõ có quần.

    Chỉ nêu TÊN BỘ PHẬN, không nêu giá trị đúng. Observer vẫn không biết chuẩn văn hoá là gì, nên vẫn
    giữ được tính khách quan; nó chỉ biết phải soi những chỗ nào.
    """
    seen, out = set(), []
    for r in contract.get("required", []):
        for w in str(r.get("description", "")).lower().replace(",", " ").split():
            w = w.strip(".,;:()")
            if w in _BO_PHAN and w not in seen:
                seen.add(w)
                out.append(w)
    return out[:toi_da]


# ------------------------------------------------------------------ A1 Visual Observer
OBSERVER_SYSTEM = (
    "You describe what is visible in a photograph. Nothing else.\n"
    "RULES, all mandatory:\n"
    "- Never say whether anything is correct, authentic, traditional, right or wrong.\n"
    "- Never name a country, culture, ethnicity or historical period.\n"
    "- Never compare the image to anything you have seen before.\n"
    "- Describe only what a camera captured: shapes, materials, how parts join, counts, colours.\n"
    "- If a detail is hidden, occluded or ambiguous, put it in uncertain_features instead of guessing.\n"
    "Write short noun phrases, 3-10 words each."
)

OBSERVER_SCHEMA = {"type": "object", "properties": {
    "subject": {"type": "string"},
    "visible_features": {"type": "array", "items": {"type": "string"}},
    "uncertain_features": {"type": "array", "items": {"type": "string"}}},
    "required": ["subject", "visible_features"]}


def observe(agent, image: str, log=print, contract: dict | None = None) -> dict:
    """A1: chỉ tả pixel. Cấm phán đúng sai — đó là việc của Critic, và tách ra mới giảm được bịa đặt.

    `contract` chỉ dùng để rút ra DANH SÁCH BỘ PHẬN cần soi (xem `_cho_can_ta`), không bao giờ đưa nội
    dung chuẩn vào đây. Quan trọng nhất là câu bắt nó nói RÕ khi một bộ phận không nhìn thấy: ở S001,
    Critic báo thiếu quần chỉ vì mô tả không khẳng định dứt khoát là có.
    """
    parts = _cho_can_ta(contract) if contract else []
    huong_dan = ("" if not parts else
                 "\nMake sure your description says something about each of these parts: "
                 + ", ".join(parts)
                 + ". For each one, if it is present say what it looks like; if it is absent or hidden, "
                   "say so explicitly, for example 'no trousers visible'. Do not say whether any of them "
                   "is correct — that is not your job.")
    d = _json(agent, OBSERVER_SYSTEM,
              "Describe this photograph." + huong_dan + "\nReturn JSON: "
              '{"subject": "..", "visible_features": [".." up to 10], "uncertain_features": [".." up to 3]}',
              OBSERVER_SCHEMA, images=[image])
    m1 = {"subject": str(d.get("subject") or "").strip(),
          "visible_features": [str(x).strip() for x in (d.get("visible_features") or []) if str(x).strip()][:10],
          "uncertain_features": [str(x).strip() for x in (d.get("uncertain_features") or []) if str(x).strip()][:3]}
    log(f"  [A1 Observer] '{m1['subject'][:44]}' · {len(m1['visible_features'])} đặc điểm nhìn thấy, "
        f"{len(m1['uncertain_features'])} không chắc")
    return m1


# ------------------------------------------------------------------ A2 Cultural Critic
CRITIC_SYSTEM = (
    "You compare an observation of a generated image against a VISUAL CONTRACT for one cultural object.\n"
    "The contract is the ONLY standard you may use. You have no other knowledge of this culture.\n"
    "RULES, all mandatory:\n"
    "- Every violation you report MUST cite a contract_id that appears in the contract given to you.\n"
    "  If you cannot cite one, do not report it.\n"
    "- Base each violation on a quote from the observation. If the observation does not mention the\n"
    f"  feature at all, it is NOT a violation — say nothing about it.\n"
    "- Report at most {MAX_VIOLATIONS} violations, the most visually important ones.\n"
    "- Also list what must be preserved: things already correct or already asked for by the prompt.\n"
    "- If nothing is violated, return an empty violations list. That is a good outcome, not a failure."
).replace("{MAX_VIOLATIONS}", str(MAX_VIOLATIONS))

CRITIC_SCHEMA = {"type": "object", "properties": {
    "violations": {"type": "array", "items": {"type": "object", "properties": {
        "contract_id": {"type": "string"}, "evidence": {"type": "string"},
        "severity": {"type": "string"}}}},
    "preserve": {"type": "array", "items": {"type": "string"}},
    "repair_priority": {"type": "array", "items": {"type": "string"}}},
    "required": ["violations"]}


def _contract_text(contract: dict) -> str:
    req = "\n".join(f"  - id={r['id']}: {r['description']}" for r in contract.get("required", []))
    con = "\n".join(f"  - id={c['id']}: {c['description']}" for c in contract.get("confusables", []))
    return (f"VISUAL CONTRACT for {contract.get('entity', '?')}\nREQUIRED (each must be visible):\n{req}"
            + (f"\nCONFUSABLE OBJECTS (must NOT be what is shown):\n{con}" if con else ""))


def critique(agent, m1: dict, contract: dict, prompt_en: str, log=print) -> dict:
    """A2: soi quan sát với contract. Chỉ được viện dẫn id CÓ THẬT — chặn bịa chuẩn văn hoá bằng máy."""
    ok_ids = {r["id"] for r in contract.get("required", [])} | {c["id"] for c in contract.get("confusables", [])}
    d = _json(agent, CRITIC_SYSTEM,
              f"{_contract_text(contract)}\n\nORIGINAL REQUEST: {prompt_en}\n\n"
              f"OBSERVATION OF THE GENERATED IMAGE:\n{json.dumps(m1, ensure_ascii=False)}\n\n"
              'Return JSON: {"violations": [{"contract_id": "..", "evidence": "..", "severity": "major|minor"}], '
              '"preserve": [".."], "repair_priority": [".."]}',
              CRITIC_SCHEMA)
    viol, bo = [], []
    for v in (d.get("violations") or []):
        cid = str(v.get("contract_id") or "").strip()
        if cid not in ok_ids:
            bo.append(cid or "(rỗng)")      # id bịa ra -> loại bằng máy, không cần tin model
            continue
        ev_txt = str(v.get("evidence") or "")[:200]
        if not _co_can_cu(ev_txt, m1):
            bo.append(f"{cid}(bằng chứng không có trong quan sát)")
            continue
        viol.append({"contract_id": cid, "evidence": ev_txt,
                     "severity": str(v.get("severity") or "major")})
    viol = viol[:MAX_VIOLATIONS]
    m2 = {"violations": viol,
          "preserve": [str(x) for x in (d.get("preserve") or [])][:5],
          "repair_priority": [str(x) for x in (d.get("repair_priority") or [])][:2]}
    log(f"  [A2 Critic] {len(viol)} lỗi: " + (", ".join(v["contract_id"] for v in viol) or "(không có)")
        + (f" · loại {len(bo)} id bịa: {bo}" if bo else ""))
    return m2


# ------------------------------------------------------------------ A3 Prompt Refiner
REFINER_SYSTEM = (
    "You write ONE short repair clause to append to an image prompt, plus a few negative terms.\n"
    "RULES, all mandatory:\n"
    "- Diffusion models have no negation. The repair clause describes what SHOULD be drawn, positively.\n"
    "  Never write 'not', 'without', 'instead of', 'no' in the clause.\n"
    "- Describe ONLY the features named in the violations. Do not redescribe the whole scene.\n"
    "- Do not introduce any object, place, colour or person that is not already in the prompt or the\n"
    "  contract. Adding new nouns changes the picture instead of repairing it.\n"
    "- The clause must be at most 25 words, one sentence.\n"
    "- negative_terms: 2 to 4 short noun phrases naming exactly the wrong things to keep out."
)

REFINER_SCHEMA = {"type": "object", "properties": {
    "repair_clause": {"type": "string"},
    "negative_terms": {"type": "array", "items": {"type": "string"}}},
    "required": ["repair_clause"]}


def refine(agent, base_prompt: str, m2: dict, contract: dict, log=print) -> dict:
    """A3: viết mệnh đề sửa. KHÔNG được viết lại cả cảnh — prompt gốc là bất biến."""
    if not m2.get("violations"):
        return {"repair_clause": "", "negative_terms": []}
    want = "\n".join(
        f"  - {v['contract_id']}: should show "
        + next((r["description"] for r in contract.get("required", []) if r["id"] == v["contract_id"]),
               next((c["description"] for c in contract.get("confusables", []) if c["id"] == v["contract_id"]), ""))
        + f"   (evidence: {v['evidence'][:90]})"
        for v in m2["violations"])
    d = _json(agent, REFINER_SYSTEM,
              f"BASE PROMPT (immutable, do not rewrite):\n{base_prompt[:600]}\n\n"
              f"MUST BE REPAIRED:\n{want}\n\nMUST BE PRESERVED: {', '.join(m2.get('preserve') or []) or '(none)'}\n\n"
              'Return JSON: {"repair_clause": "..", "negative_terms": [".."]}',
              REFINER_SCHEMA, max_new_tokens=300)
    clause = " ".join(str(d.get("repair_clause") or "").split())
    neg = [" ".join(str(x).split()) for x in (d.get("negative_terms") or []) if str(x).strip()][:4]
    # chặn phủ định: mô hình khuếch tán vẽ MỌI từ trong prompt, kể cả từ sau chữ "không"
    low = set(clause.lower().replace(",", " ").split())
    if low & {"not", "no", "without", "instead", "never", "avoid"}:
        log(f"  [A3 Refiner] mệnh đề còn phủ định -> bỏ: {clause[:60]}")
        clause = ""
    neg = _bo_negative_pha_prompt(neg, base_prompt, contract, log)
    m3 = {"repair_clause": clause, "negative_terms": neg}
    log(f"  [A3 Refiner] '{clause[:70]}' · negative {neg}")
    return m3


# ------------------------------------------------------------------ A2 duyệt lại, đúng một lần
REVIEW_SYSTEM = (
    "You check whether a proposed repair clause actually covers the violations you reported, and whether\n"
    "it contradicts anything that must be preserved.\n"
    "Approve unless there is a concrete problem. Be specific about what is wrong if you do not approve."
)

REVIEW_SCHEMA = {"type": "object", "properties": {
    "approved": {"type": "boolean"},
    "covered_contract_ids": {"type": "array", "items": {"type": "string"}},
    "contradictions": {"type": "array", "items": {"type": "string"}}},
    "required": ["approved"]}


def review(agent, m3: dict, m2: dict, contract: dict, log=print) -> dict:
    """A2 duyệt M3. Đúng MỘT lần, không tranh luận vô hạn."""
    if not m3.get("repair_clause"):
        return {"approved": False, "covered_contract_ids": [], "contradictions": ["không có mệnh đề sửa"]}
    d = _json(agent, REVIEW_SYSTEM,
              f"{_contract_text(contract)}\n\nVIOLATIONS YOU REPORTED:\n{json.dumps(m2['violations'], ensure_ascii=False)}\n"
              f"MUST BE PRESERVED: {', '.join(m2.get('preserve') or []) or '(none)'}\n\n"
              f"PROPOSED REPAIR CLAUSE: {m3['repair_clause']}\nPROPOSED NEGATIVE TERMS: {m3['negative_terms']}\n\n"
              'Return JSON: {"approved": true/false, "covered_contract_ids": [".."], "contradictions": [".."]}',
              REVIEW_SCHEMA, max_new_tokens=300)
    r = {"approved": bool(d.get("approved")),
         "covered_contract_ids": [str(x) for x in (d.get("covered_contract_ids") or [])],
         "contradictions": [str(x) for x in (d.get("contradictions") or [])][:3]}
    log(f"  [A2 duyệt] {'ĐỒNG Ý' if r['approved'] else 'TỪ CHỐI'}"
        + (f" · mâu thuẫn: {r['contradictions']}" if r["contradictions"] else ""))
    return r


# ------------------------------------------------------------------ ghép prompt
def append_repair(base_prompt: str, clause: str, max_tokens: int = MAX_TOKENS) -> tuple[str, str]:
    """Đặt mệnh đề sửa lên TRƯỚC prompt gốc. Trả (prompt đầy đủ, ghi chú).

    Vì sao đặt trước chứ không nối sau, dù tên hàm là "append":

    1. Nối sau thì KHÔNG CÒN CHỖ. Prompt Culture-TRIP dài 97-324 từ; với trần 100 token thì mọi mệnh đề
       sửa đều bị từ chối và cả ba nhánh T/S/M rơi về prompt gốc — bốn ảnh giống hệt nhau, thí nghiệm ra
       con số không. Đo thật ở lượt chạy thử S001: "prompt gốc đã 97 từ, không còn chỗ".
    2. Nối sau thì BỊ LOÃNG. compel ghép prompt dài theo từng khối 77 token, nhưng embedding gộp vẫn cắt
       ở 77. Phần đuôi gần như không tác dụng — đã thấy ở S002: câu tinh chỉnh tả rõ "a balanced pole
       across her shoulders" mà ảnh vẫn ra xe đạp.

    Prompt gốc vẫn BẤT BIẾN đúng nghĩa: không sửa, không cắt một chữ nào của nó. Chỉ có mệnh đề sửa bị
    giới hạn độ dài, và nếu nó vượt `max_clause` thì cắt MỆNH ĐỀ.
    """
    clause = " ".join((clause or "").split())
    if not clause:
        return base_prompt, "không có mệnh đề sửa"
    max_clause = 25
    cw = clause.split()
    note = ""
    if len(cw) > max_clause:
        clause = " ".join(cw[:max_clause]).rstrip(",.;") + "."
        note = f"cắt mệnh đề từ {len(cw)} còn {max_clause} từ"
    if not clause.endswith((".", ",")):
        clause += "."
    return f"{clause} {base_prompt}", note


# ------------------------------------------------------------------ nhánh S: một VLM tự viết
SINGLE_SYSTEM = (
    "You look at a generated image and the prompt that made it, and you write ONE short clause to append\n"
    "to that prompt so the next generation is a more faithful depiction of the Vietnamese cultural object.\n"
    "Diffusion models have no negation: the clause describes what SHOULD be drawn. At most 25 words.\n"
    "Also give 2 to 4 negative terms naming the wrong things to keep out."
)


def single_agent(agent, image: str, base_prompt: str, contract: dict | None, log=print) -> dict:
    """Nhánh S: KHÔNG phân vai, một lượt gọi duy nhất nhìn thẳng ảnh và viết mệnh đề.

    Đây là đối chứng quan trọng nhất của bài. Thiếu nó thì dù M thắng B cũng chỉ chứng minh được
    "nhìn ảnh có ích", không chứng minh được "phân vai có ích", và sơ đồ ba agent chỉ là trang trí.
    Nó ĐƯỢC nhận contract để so sánh công bằng với M; khác biệt duy nhất là không tách vai.
    """
    d = _json(agent, SINGLE_SYSTEM,
              (f"{_contract_text(contract)}\n\n" if contract else "")
              + f"PROMPT THAT MADE THIS IMAGE:\n{base_prompt[:600]}\n\n"
                'Return JSON: {"repair_clause": "..", "negative_terms": [".."]}',
              REFINER_SCHEMA, images=[image], max_new_tokens=300)
    clause = " ".join(str(d.get("repair_clause") or "").split())
    low = set(clause.lower().replace(",", " ").split())
    if low & {"not", "no", "without", "instead", "never", "avoid"}:
        clause = ""
    neg = [" ".join(str(x).split()) for x in (d.get("negative_terms") or [])][:4]
    if contract:
        neg = _bo_negative_pha_prompt(neg, base_prompt, contract, log)
    m3 = {"repair_clause": clause, "negative_terms": neg}
    log(f"  [S một agent] '{clause[:70]}' · negative {neg}")
    return m3


# ------------------------------------------------------------------ nhánh T: không nhìn ảnh
def text_only(agent, base_prompt: str, contract: dict, log=print) -> dict:
    """Nhánh T: Refiner có contract nhưng KHÔNG nhìn ảnh. Tách "phản hồi thị giác" khỏi "prompt dài hơn".

    Không có nhánh này thì M thắng B cũng có thể chỉ vì prompt được nối thêm chữ, chẳng liên quan gì
    tới việc hệ thống đã nhìn thấy ảnh.
    """
    gia_dinh = {"violations": [{"contract_id": r["id"], "evidence": "(không nhìn ảnh)", "severity": "major"}
                               for r in contract.get("required", [])[:MAX_VIOLATIONS]],
                "preserve": [], "repair_priority": []}
    m3 = refine(agent, base_prompt, gia_dinh, contract, log=log)
    log("  [T không nhìn ảnh] dùng đúng contract, giả định mọi mục required đều thiếu")
    return m3


# ------------------------------------------------------------------ điều phối nhánh M
def run_multi(agent, image: str, base_prompt: str, contract: dict, prompt_id: str = "", log=print) -> Trace:
    """Observer -> Critic -> Refiner -> Critic duyệt. Bất kỳ bước nào hỏng -> no-op, trả prompt gốc.

    No-op KHÔNG phải thất bại: nó khiến nhánh M trả về đúng ảnh nền với đúng seed, nên M không bao giờ
    tệ hơn B. Vòng lặp cũ làm ảnh TỆ ĐI ở 2/3 prompt chính vì thiếu đường lùi này.
    """
    t = Trace(prompt_id=prompt_id, arm="M")
    t.m1 = observe(agent, image, log, contract)
    if not t.m1.get("visible_features"):
        t.ly_do_noop = "Observer không mô tả được gì"
        return t
    t.m2 = critique(agent, t.m1, contract, base_prompt, log)
    if not t.m2.get("violations"):
        t.ly_do_noop = "Critic không tìm thấy lỗi nào viện dẫn được contract"
        return t
    t.m3 = refine(agent, base_prompt, t.m2, contract, log)
    if not t.m3.get("repair_clause"):
        t.ly_do_noop = "Refiner không viết được mệnh đề hợp lệ"
        return t
    t.review = review(agent, t.m3, t.m2, contract, log)
    if not t.review.get("approved"):
        t.ly_do_noop = "Critic từ chối đề xuất: " + "; ".join(t.review.get("contradictions") or ["không nêu lý do"])
        return t
    t.repair_clause = t.m3["repair_clause"]
    t.negative_terms = t.m3.get("negative_terms") or []
    t.noop = False
    return t
