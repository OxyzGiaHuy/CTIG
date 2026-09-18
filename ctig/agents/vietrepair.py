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
#: Ngưỡng dừng, thang 0-10, lấy theo T2I-Copilot (ICCV'25): điểm > 8,0 là xong, không cần đủ điểm tối đa.
#: Bắt phải đủ tối đa là lý do vòng lặp chạy hết số vòng rồi làm hỏng ảnh đã đúng.
NGUONG_DAT = 8.0
#: Mục dưới mức này coi như hỏng, đưa cho Refiner. 5/10 = "có nhưng sai nửa".
NGUONG_HONG = 5.0

ROOT = Path(__file__).resolve().parent.parent.parent


def load_contracts(path: str | Path | None = None) -> dict:
    # Prefer the visibility-aware v2 artifact.  The legacy keys are preserved in
    # that file, so old callers remain compatible.  An explicit path always wins.
    p = Path(path) if path else ROOT / "data" / "contracts_v2.json"
    if not path and not p.exists():
        p = ROOT / "data" / "contracts.json"
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


def _cho_can_ta(contract: dict, toi_da: int = 8) -> list[str]:
    """Rút danh sách BỘ PHẬN cần mô tả từ contract, đã bóc hết đáp án.

    Vì sao cần: ở S001, Observer trả về danh sách cụm rời rạc ('white pants', 'white top with red
    patterns'). Contract hỏi "áo có mặc TRÊN quần riêng không" — một QUAN HỆ giữa hai vật, mà danh sách
    rời rạc không nói được. Critic phải suy diễn và suy sai: nó báo thiếu quần trong khi Observer đã ghi
    rõ có quần.

    Chỉ nêu TÊN BỘ PHẬN, không nêu giá trị đúng. Observer vẫn không biết chuẩn văn hoá là gì, nên vẫn
    giữ được tính khách quan; nó chỉ biết phải soi những chỗ nào.

    Ưu tiên trường `part` viết tay trong contract; không có thì mới dò từ `description` bằng `_BO_PHAN`.
    Vì sao phải có `part`: bản dò tự động BỎ SÓT 19/65 mục — `scallion_and_onion_on_top`,
    `gourd_cup_on_rod`, `curtain_hides_operators`, `open_front_two_flaps`, ... đều không chứa từ nào
    trong `_BO_PHAN`, nên Observer không bao giờ được chỉ đi soi chỗ đó, Critic không bao giờ thấy nó
    trong quan sát, và mục đó thành CHẾT: có nằm trong contract cũng như không. Đã thấy hậu quả ở S012,
    nơi mệnh đề sửa bàn về mái chèo thay vì cái thân thuyền tròn. Viết tay `part` là cách chỉnh hệ
    thống theo từng prompt mà không phải đụng vào mã; nhiều bộ phận cho một mục thì ngăn bằng "|".
    """
    seen, out = set(), []
    for r in contract.get("required", []):
        # v2: a factual-but-unobservable item may be retained for provenance while
        # explicitly excluded from judging and repair.
        if r.get("scoring") == "excluded" or r.get("visibility") == "optional":
            continue
        tay = str(r.get("part") or "").strip().lower()
        if tay:
            for w in tay.split("|"):
                w = " ".join(w.split())
                if w and w not in seen:
                    seen.add(w)
                    out.append(w)
            continue
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
    required = [r for r in contract.get("required", [])
                if r.get("scoring", "required") == "required"]
    conditional = [r for r in contract.get("required", [])
                   if r.get("scoring") == "conditional"]
    req = "\n".join(
        f"  - id={r['id']} [priority={r.get('importance', 3)}]: {r['description']}"
        for r in sorted(required, key=lambda x: -int(x.get("importance", 3)))
    )
    cond = "\n".join(
        f"  - id={r['id']} [priority={r.get('importance', 2)}]: {r['description']}"
        for r in conditional
    )
    con = "\n".join(f"  - id={c['id']}: {c['description']}" for c in contract.get("confusables", []))
    return (f"VISUAL CONTRACT for {contract.get('entity', '?')}\nREQUIRED VISIBLE EVIDENCE:\n{req}"
            + ("\nCONDITIONAL (judge only when that part is visible; absence/occlusion is not a violation):\n"
               + cond if cond else "")
            + (f"\nCONFUSABLE OBJECTS (must NOT be what is shown):\n{con}" if con else ""))


def critique(agent, m1: dict, contract: dict, prompt_en: str, log=print) -> dict:
    """A2: soi quan sát với contract. Chỉ được viện dẫn id CÓ THẬT — chặn bịa chuẩn văn hoá bằng máy."""
    ok_ids = {r["id"] for r in contract.get("required", [])
              if r.get("scoring", "required") != "excluded"} | \
             {c["id"] for c in contract.get("confusables", [])}
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
    "- NEVER change how the picture is framed. Do not mention close-up, portrait, wide shot, full body,\n"
    "  top-down, flat lay, overhead, zoom, crop, angle, composition, lighting or background. Write only\n"
    "  about the object itself. Changing the framing is the most common way a repair makes things worse.\n"
    "- negative_terms: 2 to 4 short noun phrases naming exactly the wrong things to keep out.\n"
    "  Negative terms must also never mention framing, composition, lighting or camera angle."
)

#: CỤM chỉ khung hình. Phải là cụm, không được là từ đơn — bản đầu lọc theo từ đơn và giết mất 12/32
#: mệnh đề sửa, toàn những mệnh đề quan trọng nhất, chỉ vì chữ "flat":
#:   "The flat white rice noodles are cut from thin sheets"   (bánh phở SỢI DẸT — đặc trưng định danh)
#:   "tied with flat bamboo strips"                           (lạt tre bánh chưng)
#:   "very wide flat-brimmed round hats"                      (nón quai thao)
#:   "a long narrow flat soundbox"                            (đàn bầu)
#: "flat" và "wide" là từ mô tả cốt lõi của ít nhất bốn trong mười sáu thực thể. Lọc theo từ đơn ở đây
#: gây hại nhiều hơn lợi.
_KHUNG_HINH = ("close-up", "close up", "closeup", "wide shot", "wide-angle", "long shot",
               "full body shot", "full-body shot", "top-down", "top down", "overhead view",
               "flat lay", "flat-lay", "bird's eye", "birds eye", "zoomed in", "zoom in",
               "camera angle", "point of view", "shallow depth", "depth of field",
               "in the background", "soft lighting", "studio lighting", "cropped to",
               "framed as", "portrait shot", "medium shot")


def _bo_khung_hinh(clause: str, neg: list[str], log=print) -> tuple[str, list[str]]:
    """Loại mọi thứ nói về khung hình khỏi mệnh đề sửa và khỏi negative.

    Nhắc trong system prompt là không đủ — mô hình vẫn viết 'shown in a wider shot'. Chặn bằng máy:
    mệnh đề chứa từ khung hình thì BỎ CẢ MỆNH ĐỀ (rơi về no-op, an toàn hơn là sửa nửa vời), còn
    negative thì chỉ bỏ cụm vi phạm.
    """
    low = " " + " ".join((clause or "").lower().replace(",", " ").split()) + " "
    hit = next((c for c in _KHUNG_HINH if c in low), None)
    if hit:
        log(f"  [khung hình] mệnh đề nói về bố cục ('{hit}') -> bỏ: {clause[:60]}")
        clause = ""
    out = []
    for phrase in neg or []:
        pl = " " + " ".join(phrase.lower().replace(",", " ").split()) + " "
        h2 = next((c for c in _KHUNG_HINH if c in pl), None)
        if h2:
            log(f"  [khung hình] bỏ negative '{phrase}' ('{h2}')")
            continue
        out.append(phrase)
    return clause, out

REFINER_SCHEMA = {"type": "object", "properties": {
    "repair_clause": {"type": "string"},
    "negative_terms": {"type": "array", "items": {"type": "string"}}},
    "required": ["repair_clause"]}


def _muc_can_sua(v: dict, contract: dict) -> str:
    """Một dòng cho Refiner. Mục REQUIRED và mục CONFUSABLE phải diễn đạt NGƯỢC NHAU.

    Lỗi đã mắc và đã thấy tận mắt ở S001: bản đầu tra mô tả theo `contract_id` ở CẢ HAI danh sách rồi
    ghép "should show <mô tả>". Với confusable thì mô tả là thứ KHÔNG ĐƯỢC CÓ, nên khi bộ rà bắt đúng
    'centre_front_frog_buttons', Refiner viết ra "The áo dài has a row of knotted cloth frog buttons
    down the middle of the chest" — tức bảo model vẽ thêm đúng cái sai vừa bắt được. Vòng sau lặp lại
    với 'qipao_cheongsam'. Đây là kiểu lỗi im lặng: ảnh vẫn sinh ra, log vẫn đẹp, chỉ có kết quả là
    ngược.
    """
    cid = v["contract_id"]
    bc = f"   (evidence: {v['evidence'][:90]})"
    r = next((x for x in contract.get("required", []) if x["id"] == cid), None)
    if r:
        return f"  - {cid}: should show {r['description']}{bc}"
    c = next((x for x in contract.get("confusables", []) if x["id"] == cid), None)
    if c:
        muon = "; ".join(x["description"] for x in contract.get("required", [])[:3])
        return (f"  - {cid}: the picture currently shows the WRONG OBJECT, namely {c['description']}. "
                f"Do not describe that. Describe the intended object: {muon}{bc}")
    return f"  - {cid}: unknown contract id{bc}"


def refine(agent, base_prompt: str, m2: dict, contract: dict, log=print) -> dict:
    """A3: viết mệnh đề sửa. KHÔNG được viết lại cả cảnh — prompt gốc là bất biến."""
    if not m2.get("violations"):
        return {"repair_clause": "", "negative_terms": []}
    want = "\n".join(_muc_can_sua(v, contract) for v in m2["violations"])
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
    ten_conf = {x["id"] for x in contract.get("confusables", [])}
    for v in m2["violations"]:
        if v["contract_id"] in ten_conf:
            t = v["contract_id"].replace("_", " ")
            if t not in neg:
                neg.append(t)          # tên, không phải mô tả: mô tả confusable hay chứa 'no/without'
    neg = _bo_negative_pha_prompt(neg[:5], base_prompt, contract, log)
    clause, neg = _bo_khung_hinh(clause, neg, log)
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
def append_repair(base_prompt: str, clause: str, orig: str | None = None,
                  max_clause: int = MAX_TOKENS) -> tuple[str, str]:
    """Chèn mệnh đề sửa NGAY SAU câu gốc, trước phần Culture-TRIP viết thêm.

    Thứ tự cuối cùng:  [câu gốc]  +  [mệnh đề sửa]  +  [phần Culture-TRIP viết thêm]

    Ba ràng buộc phải thoả cùng lúc, và chỉ vị trí này thoả cả ba:

    1. **Câu gốc phải đứng đầu.** Đó là ý định của người dùng, và Culture-TRIP vốn đã ghép theo thứ tự
       [gốc] + [viết thêm] ở 99/100 file (cờ `composed`). Bản trước của hàm này chèn mệnh đề sửa lên
       TRƯỚC cả cụm đó, tức đẩy câu gốc xuống hàng hai — phá đúng luật ấy.
    2. **Mệnh đề sửa phải nằm trong 77 token đầu.** compel ghép prompt dài theo từng khối 77 token nhưng
       embedding gộp vẫn cắt ở 77, nên phần đuôi gần như không tác dụng. Nối vào cuối thì mệnh đề vô
       hiệu — đã thấy ở S002: câu tả rõ "a balanced pole across her shoulders" mà vẫn ra xe đẩy.
    3. **Prompt gốc không bị cắt một chữ nào.** Chỉ mệnh đề sửa bị giới hạn độ dài.

    Cái bị đẩy lùi là phần Culture-TRIP viết thêm — và đó đúng là phần NÊN bị đẩy lùi, vì chính nó sinh
    ra hoa văn Trung Quốc ở S001 ("intricate patterns embroidered on the front and back panels").

    `orig` là câu gốc; không truyền hoặc không khớp đầu `base_prompt` thì lùi về chèn lên trước.
    """
    clause = " ".join((clause or "").split())
    if not clause:
        return base_prompt, "không có mệnh đề sửa"
    cw = clause.split()
    note = ""
    if len(cw) > max_clause:
        clause = " ".join(cw[:max_clause]).rstrip(",.;") + "."
        note = f"cắt mệnh đề từ {len(cw)} còn {max_clause} từ"
    if not clause.endswith((".", ",")):
        clause += "."
    o = " ".join((orig or "").split())
    if o and base_prompt.strip().startswith(o):
        con_lai = base_prompt.strip()[len(o):].lstrip()
        return f"{o} {clause} {con_lai}".strip(), note
    return f"{clause} {base_prompt}", (note + "; câu gốc không khớp đầu prompt -> chèn lên trước").strip("; ")


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
    clause, neg = _bo_khung_hinh(clause, neg, log)
    m3 = {"repair_clause": clause, "negative_terms": neg}
    log(f"  [S một agent] '{clause[:70]}' · negative {neg}")
    return m3


# ------------------------------------------------------------------ nhánh T: không nhìn ảnh
def text_only(agent, base_prompt: str, contract: dict, log=print) -> dict:
    """Nhánh T: Refiner có contract nhưng KHÔNG nhìn ảnh. Tách "phản hồi thị giác" khỏi "prompt dài hơn".

    Không có nhánh này thì M thắng B cũng có thể chỉ vì prompt được nối thêm chữ, chẳng liên quan gì
    tới việc hệ thống đã nhìn thấy ảnh.
    """
    active = [r for r in contract.get("required", [])
              if r.get("scoring", "required") == "required"]
    active.sort(key=lambda r: -int(r.get("importance", 3)))
    gia_dinh = {"violations": [{"contract_id": r["id"], "evidence": "(không nhìn ảnh)", "severity": "major"}
                               for r in active[:MAX_VIOLATIONS]],
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

# ==================================================================== bản rút gọn kiểu T2I-Copilot
# Critic tự do viết `violations` có hai chỗ yếu đo được: nó chỉ nêu tối đa 2 lỗi nên phần lớn contract
# không bao giờ được rà, và nó gần như không bao giờ tự nhắc tới `confusables` (nhánh M no-op 53%, lý do
# áp đảo là Refiner không có gì hợp lệ để viết). Ở đây đổi sang HỎI TỪNG MỤC một câu có/không trên MỘT
# ảnh với lựa chọn bằng chữ — đúng dạng câu hỏi mà Mistral làm được: đo ngày 2026-09-17, nó tin cậy ở
# dạng một-ảnh-cộng-lựa-chọn nhưng thiên lệch vị trí 0,42 ở dạng so hai ảnh.
#
# CẢNH BÁO PHƯƠNG PHÁP, đừng quên khi viết bài: `diem()` tính TỪ contract, mà vòng lặp lại tối ưu thẳng
# vào nó. Nên "điểm sau >= điểm trước" là ĐỒNG NHẤT THỨC, không phải kết quả — đúng cái lỗi đã giết vòng
# lặp ba lượt cũ. Điểm này chỉ được dùng để LÁI vòng lặp và để dừng sớm; kết luận của bài phải đứng trên
# nhãn người, VQAScore theo prompt gốc, và tương đồng với ảnh thật cất riêng.

CHECK_SYSTEM = (
    "You rate how well ONE photograph matches ONE written statement. Nothing else.\n"
    "RULES, all mandatory:\n"
    "- Rate only the statement you are given, against what is visible in this photograph.\n"
    "- score is 0 to 10.  10 = exactly as described.  8 = as described, minor deviation.\n"
    "  5 = the thing is there but clearly differs from the description.\n"
    "  2 = something else is in its place.  0 = nothing of the kind is there.\n"
    "- Do not give a high score because the picture looks good, or because the object is\n"
    "  roughly of the right family. Rate the specific thing the statement describes.\n"
    "- If the part the statement is about lies OUTSIDE the picture — cut off by the frame,\n"
    "  below or beyond the edge — set in_frame to false and do not guess a score.\n"
    "- Never name a country, culture or ethnicity in your answer.\n"
    "- evidence: what you actually see at that place, at most 15 words."
)

CHECK_SCHEMA = {"type": "object", "properties": {
    "in_frame": {"type": "boolean"}, "score": {"type": "number"},
    "evidence": {"type": "string"}}, "required": ["in_frame"]}


def _hoi_mot_muc(agent, image: str, cau: str, part: str) -> dict:
    """Chấm MỘT mục contract trên thang 0-10.

    Vì sao bỏ có/không: đo trên S001 ngày 2026-09-18, một tấm áo sơ mi trắng với quần ống rộng và một
    tấm áo dài thật cùng được tính "có tà dài" nên hoà 3/4, tức thang nhị phân không phân biệt được hai
    ảnh khác hẳn nhau. Thang 0-10 là luật của T2I-Copilot (ICCV'25) và cho vòng lặp một hướng dốc để đi.
    """
    noi = f"Look closely at the {part} of the main subject.\n" if part else ""
    d = _json(agent, CHECK_SYSTEM,
              f"{noi}Statement: \"{cau}\"\n"
              "How fully does this photograph match that statement?\n"
              'Return JSON: {"in_frame": true|false, "score": 0-10, "evidence": ".."}',
              CHECK_SCHEMA, images=[image], max_new_tokens=120)
    trong = bool(d.get("in_frame", True))
    try:
        sc = max(0.0, min(10.0, float(d.get("score"))))
    except (TypeError, ValueError):
        sc, trong = 0.0, False       # không đọc được điểm -> coi như không phán được, KHÔNG đoán bừa
    return {"in_frame": trong, "score": sc if trong else None,
            "evidence": " ".join(str(d.get("evidence") or "").split())[:120]}


CHON_SYSTEM = (
    "You are shown one photograph and a numbered list of object descriptions.\n"
    "Choose the ONE description that best matches the main subject as a WHOLE.\n"
    "RULES, all mandatory:\n"
    "- Judge the whole object, not single words. A description that matches only part of what you see\n"
    "  is the WRONG answer.\n"
    "- Never name a country, culture or ethnicity in your answer.\n"
    "- If none of them matches the main subject, choose 0.\n"
    "- evidence: the one detail that decided it, at most 15 words."
)

CHON_SCHEMA = {"type": "object", "properties": {
    "choice": {"type": "integer"}, "evidence": {"type": "string"}}, "required": ["choice"]}


def _chon_vat(agent, image: str, contract: dict, log=print) -> dict:
    """Hỏi MỘT câu trắc nghiệm: vật trong ảnh giống thứ đúng hay giống một confusable?

    Thay cho ba câu có/không riêng lẻ. Lý do đổi, đo trên S001 ngày 2026-09-18: hỏi riêng từng
    confusable thì model khớp lẻ tẻ vài từ trong mô tả dài rồi gật, và bằng chứng nó tự viết lại nói
    ngược với phán quyết của chính nó — 'hanbok: yes' kèm bằng chứng "no skirt visible", 'qipao: yes'
    kèm bằng chứng "wide trousers visible" trong khi mô tả qipao ghi rõ là KHÔNG có quần rời. Vòng lặp
    đuổi theo confusable ma suốt 4 vòng và điểm tụt từ 2/4 xuống 1/4.

    Trắc nghiệm ép chọn MỘT, nên khớp lẻ tẻ không thể làm cháy hết mọi lựa chọn cùng lúc. Đây cũng
    đúng dạng câu hỏi đã đo là bộ chấm làm được: một ảnh + lựa chọn bằng chữ.

    Thứ tự lựa chọn xáo theo băm đường dẫn ảnh, để thiên lệch vị trí không dồn hết vào một phía.
    """
    import hashlib

    active = [r for r in contract.get("required", [])
              if r.get("scoring", "required") == "required"]
    active.sort(key=lambda r: -int(r.get("importance", 3)))
    dung = "; ".join(r["description"] for r in active[:3])
    ds = [("__dung__", f"{contract.get('entity', 'the intended object')}: {dung}")] + \
         [(x["id"], x["description"]) for x in contract.get("confusables", [])]
    k = int(hashlib.sha1(str(image).encode()).hexdigest(), 16)
    ds = ds[k % len(ds):] + ds[:k % len(ds)]          # xoay vòng, lặp lại được
    hoi = "\n".join(f"  {i + 1}. {t}" for i, (_, t) in enumerate(ds))
    d = _json(agent, CHON_SYSTEM,
              f"Which ONE of these best describes the main subject of this photograph?\n{hoi}\n"
              f"  0. none of these\n"
              'Return JSON: {"choice": <number>, "evidence": ".."}',
              CHON_SCHEMA, images=[image], max_new_tokens=120)
    try:
        i = int(d.get("choice"))
    except (TypeError, ValueError):
        i = 0
    cid = ds[i - 1][0] if 1 <= i <= len(ds) else "__khong__"
    ev = " ".join(str(d.get("evidence") or "").split())[:120]
    log(f"  [chọn vật] -> {cid} · {ev}")
    return {"chon": cid, "evidence": ev, "thu_tu": [x for x, _ in ds]}


def kiem_tung_muc(agent, image: str, contract: dict, log=print) -> dict:
    """Rà toàn bộ contract: mỗi mục required một câu chấm 0-10, cộng MỘT câu trắc nghiệm confusable.

    Điểm của ảnh = trung bình có trọng số importance của các mục PHÁN ĐƯỢC
    (mục ngoài khung bị loại khỏi mẫu số, không bị tính 0; mục excluded không được hỏi).
    Lẫn confusable là CỬA CHẶN chứ không phải trừ điểm: còn lẫn thì không bao giờ được coi là đạt, dù
    điểm trung bình có cao đến đâu — sinh ra nhầm hẳn vật khác không thể bù bằng chi tiết đúng.
    """
    bang, thieu = [], []
    active_required = [r for r in contract.get("required", [])
                       if r.get("scoring", "required") != "excluded"]
    for r in active_required:
        d = _hoi_mot_muc(agent, image, r["description"], str(r.get("part") or ""))
        importance = max(1, min(3, int(r.get("importance", 1))))
        bang.append({"contract_id": r["id"], "loai": "required",
                     "importance": importance, **d})
        if d["in_frame"] and d["score"] < NGUONG_HONG:
            thieu.append({"contract_id": r["id"], "evidence": d["evidence"],
                          "score": d["score"],
                          "severity": "major" if importance == 3 else "minor",
                          "importance": importance})

    ch = _chon_vat(agent, image, contract, log)
    lan = []
    for x in contract.get("confusables", []):
        trung = ch["chon"] == x["id"]
        bang.append({"contract_id": x["id"], "loai": "confusable",
                     "verdict": "yes" if trung else "no", "evidence": ch["evidence"] if trung else ""})
        if trung:
            lan.append({"contract_id": x["id"], "evidence": ch["evidence"], "severity": "major"})
    bang.append({"contract_id": "__chon__", "loai": "chon", "verdict": ch["chon"],
                 "evidence": ch["evidence"], "thu_tu": ch["thu_tu"]})

    req = [b for b in bang if b["loai"] == "required"]
    ngoai = [b["contract_id"] for b in req if not b["in_frame"]]
    phan_duoc = [b for b in req if b["in_frame"]]
    tong_trong_so = sum(b["importance"] for b in phan_duoc)
    diem = (round(sum(b["score"] * b["importance"] for b in phan_duoc) / tong_trong_so, 2)
            if tong_trong_so else 0.0)
    thieu.sort(key=lambda t: (-t["importance"], t["score"]))  # ưu tiên lỗi định danh, rồi lỗi nặng
    ket = {"bang": bang, "thieu": thieu, "lan": lan, "diem": diem,
           "so_phan_duoc": len(phan_duoc), "so_required": len(active_required),
           "ngoai_khung": ngoai, "dat": diem >= NGUONG_DAT and not lan}
    log(f"  [rà contract] điểm {diem:.1f}/10 trên {len(phan_duoc)} mục"
        f"{' (ngoài khung: ' + ', '.join(ngoai) + ')' if ngoai else ''}"
        + "  " + " ".join(f"{b['contract_id']}={b['score']:.0f}" for b in phan_duoc)
        + (f" · LẪN {[l['contract_id'] for l in lan]}" if lan else "")
        + ("  -> ĐẠT" if ket["dat"] else ""))
    return ket


def _m2_tu_check(ket: dict, contract: dict) -> dict:
    """Đổi bảng rà thành đúng dạng `m2` mà `refine()` đã nhận, để không phải viết lại Refiner.

    Cái LẪN xếp trước cái THIẾU: sinh ra nhầm hẳn vật khác là hỏng nặng hơn là thiếu một chi tiết.
    """
    dat = {b["contract_id"] for b in ket["bang"]
           if b["loai"] == "required" and b.get("in_frame") and b["score"] >= NGUONG_DAT}
    return {"violations": (ket["lan"] + ket["thieu"])[:MAX_VIOLATIONS],
            "preserve": [next(r["description"] for r in contract["required"] if r["id"] == i)
                         for i in list(dat)[:5]],
            "repair_priority": [v["contract_id"] for v in ket["lan"] + ket["thieu"]]}



def run_loop(agent, sinh, i0: str, base_prompt: str, contract: dict, orig_en: str = "",
             prompt_id: str = "", max_vong: int = 4, kien_nhan: int = 3, log=print) -> dict:
    """Vòng lặp rà-sửa-sinh lại, dừng khi đủ required và hết lẫn confusable, hoặc khi chững.

    `sinh(prompt_terms, negative, sub, dung_ref) -> đường dẫn ảnh` do người gọi cung cấp và PHẢI giữ NGUYÊN SEED
    qua mọi vòng: chỉ câu prompt được đổi. Đổi seed thì mỗi vòng là một lần bốc thăm mới, và "vòng lặp
    hơn nhánh nền" sẽ chỉ là chuyện sinh nhiều rồi chọn — đã đo: best-of-4 bốc thăm thắng vòng lặp cũ ở
    2/3 prompt khi cùng ngân sách.

    Mệnh đề sửa mỗi vòng THAY THẾ mệnh đề vòng trước, không cộng dồn. Cộng dồn thì 3 vòng × 25 từ vượt
    77 token CLIP, phần đuôi thành vô tác dụng, mà đó lại đúng là phần vừa viết.

    Từ vòng 1 trở đi sinh KÈM ẢNH THẬT qua IP-Adapter (`dung_ref=True`), đúng như lô pilotC/loopC2 đã
    làm. Lý do có số: ở lô đó iter0 chạy không ref cho ra áo hoa văn Trung Quốc, còn iter1-3 chạy
    `sdxl_base_ref` với 3 ảnh `selected/S001/` cho ra áo dài trắng đúng chuẩn. Thứ sửa được ảnh là ẢNH
    THẬT, không phải lời phê bình — nên bỏ nó đi là bỏ mất cần gạt mạnh nhất.

    Vòng 0 KHÔNG ref, để nó trùng đúng nhánh nền B và hiệu số đo được. Và vì vòng lặp nay dùng ảnh thật,
    nhánh R (ảnh thật, không agent) trở thành mốc BẮT BUỘC: thiếu R thì mọi cải thiện đều quy về ảnh
    thật được, không nói được gì về agent.

    Trả về ảnh có điểm cao nhất; hoà điểm thì lấy vòng SỚM NHẤT, vì càng sửa càng xa ảnh gốc.
    """
    lich_su, anh, clause, neg = [], i0, "", []
    tot_nhat, chung = -99, 0
    ly_do = "hết số vòng"
    for vong in range(max_vong):
        ket = kiem_tung_muc(agent, anh, contract, log)
        lich_su.append({"vong": vong, "anh": anh, "clause": clause, "negative": neg,
                        "diem": ket["diem"], "dat": ket["dat"], "so_phan_duoc": ket["so_phan_duoc"],
                        "so_required": ket["so_required"], "ngoai_khung": ket["ngoai_khung"],
                        "thieu": [t["contract_id"] for t in ket["thieu"]],
                        "lan": [l["contract_id"] for l in ket["lan"]], "bang": ket["bang"]})
        if ket["diem"] > tot_nhat:
            tot_nhat, chung = ket["diem"], 0
        else:
            chung += 1
        if ket["so_phan_duoc"] == 0 and not ket["lan"]:
            ly_do = ("không mục required nào phán được trong khung này ("
                     + ", ".join(ket["ngoai_khung"]) + ") -> contract lệch khung hình, không phải lỗi agent")
            break
        if ket["dat"]:
            ly_do = f"điểm {ket['diem']:.1f} >= ngưỡng {NGUONG_DAT}, không lẫn confusable"
            break
        if chung >= kien_nhan:
            ly_do = f"{kien_nhan} vòng liền không khá hơn"
            break
        if vong == max_vong - 1:
            break
        m3 = refine(agent, base_prompt, _m2_tu_check(ket, contract), contract, log)
        if not m3.get("repair_clause"):
            ly_do = "Refiner không viết được mệnh đề hợp lệ"
            break
        clause, neg = m3["repair_clause"], m3.get("negative_terms") or []
        full, _ = append_repair(base_prompt, clause, orig_en)
        moi = sinh([full], neg, f"vong{vong + 1}", True)
        if not moi:
            ly_do = "sinh ảnh hỏng"
            break
        anh = moi

    # T2I-Copilot hết vòng thì trả ảnh MỚI NHẤT. Ở đây trả ảnh TỐT NHẤT: đã đo được ảnh vòng cuối tệ
    # hơn hẳn vòng 0 (S001, vòng lặp bỏ mất tà áo dài mà điểm nhị phân vẫn tăng).
    best = min(lich_su, key=lambda h: (bool(h["lan"]), -h["diem"], h["vong"]))
    log(f"  [vòng lặp] {len(lich_su)} vòng · dừng vì {ly_do} · "
        f"chọn vòng {best['vong']} điểm {best['diem']:.1f}/10"
        + (f" (còn lẫn {best['lan']})" if best["lan"] else ""))
    return {"prompt_id": prompt_id, "anh": best["anh"], "vong_chon": best["vong"],
            "diem": best["diem"], "dat": best["dat"], "lan": best["lan"],
            "diem_dau": lich_su[0]["diem"], "so_required": best["so_required"],
            "clause": best["clause"], "negative": best["negative"],
            "so_vong": len(lich_su), "ly_do_dung": ly_do, "lich_su": lich_su}
