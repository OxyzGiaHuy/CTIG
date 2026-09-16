"""
Knowledge base văn hoá Việt.

Trong hệ thống thật, chỗ này là một vector store + Wikipedia + kho ảnh có kiểm
duyệt. Ở đây nó là một file JSON để bạn thấy rõ *hình dạng* của bằng chứng cần
có. Trường quan trọng nhất không phải mô tả, mà là ba trường:

    must_have        - vẽ thiếu là sai
    must_not         - vẽ có là sai
    confusable_with  - cái mà model sẽ vẽ ra khi nó không biết

Trường thứ ba là mấu chốt: một hệ sinh ảnh văn hoá không thất bại bằng cách
vẽ ra thứ vô nghĩa, nó thất bại bằng cách vẽ ra thứ *của văn hoá khác*.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path


def normalize(text: str) -> str:
    """Bỏ dấu, hạ chữ thường.

    CHỈ dùng cho so khớp mờ theo token (xem _covered trong llm/mock.py).
    KHÔNG dùng để tìm thuật ngữ trong câu - dùng contains() bên dưới.
    """
    decomposed = unicodedata.normalize("NFD", text.lower())
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return stripped.replace("đ", "d").replace("Đ", "d")


def contains(text: str, term: str) -> bool:
    """Thuật ngữ `term` có xuất hiện trong `text` như một từ trọn vẹn không.

    Đây là hàm sửa một lỗi thật đã xảy ra trong lần chạy đầu: phiên bản cũ khớp
    chuỗi con trên text ĐÃ BỎ DẤU, nên:

        "Khuê Văn Các"  -> bỏ dấu thành "khue van cac", chứa "hue"
                        -> hệ suy ra bối cảnh Huế / Trung Bộ
                        -> loại Văn Miếu (Hà Nội) vì "xung đột vùng"
                        -> CulturalSpec rỗng -> vòng review không có gì để kiểm
                        -> báo "ĐẠT" trong khi chẳng kiểm chứng được gì.

    Một lỗi khớp chuỗi ở stage 1 lan thành con số pass_rate sai ở stage 6. Đây là
    lý do mỗi stage phải ghi log riêng: không có stage2/stage3 trên đĩa thì gần
    như không thể truy ra nguồn.

    Cách sửa: khớp theo biên từ, trên text CÒN DẤU. "phở" vẫn khớp "phở bò";
    "pho" không còn khớp "phố"; input không dấu "pho bo" vẫn khớp alias "pho".
    """
    if not term or len(term) < 2:
        return False
    # (?<!\w) / (?!\w) thay cho \b vì \b không đáng tin với ký tự có dấu.
    pattern = rf"(?<!\w){re.escape(term.lower())}(?!\w)"
    return re.search(pattern, text.lower()) is not None


def tokens(text: str) -> set[str]:
    """Token đã bỏ dấu, bỏ token quá ngắn (dưới 3 ký tự)."""
    return {t for t in normalize(text).split() if len(t) > 2}


def attr_covered(
    required: str, observed: list[str], exclude: frozenset[str] | set[str] = frozenset()
) -> bool:
    """Thuộc tính `required` có được thể hiện trong danh sách quan sát không.

    So khớp theo tỉ lệ token trùng nhau, không so nguyên văn - vì caption của một
    VLM thật sẽ không bao giờ trùng từng chữ với bằng chứng.

    `exclude` loại bỏ các token KHÔNG mang tính phân biệt trước khi so. Tham số
    này sinh ra từ một lỗi thật, và nó minh hoạ giới hạn cốt lõi của mọi cách
    so khớp kiểu túi-từ:

        bắt buộc:  "nước dùng trong, màu hổ phách nhạt, KHÔNG sánh đặc"
        bị cấm:    "nước dùng đỏ sánh cay kiểu mì Tứ Xuyên"

    Hai câu này nghĩa TRÁI NGƯỢC nhau, nhưng chia sẻ {nước, dùng, sánh} = 50%
    token, đủ vượt ngưỡng. Kết quả: hệ báo "Phở mang chi tiết bị cấm: nước dùng
    đỏ sánh cay" trên một tô phở được vẽ hoàn toàn đúng, rồi lặp thêm ba vòng để
    sửa một lỗi không tồn tại.

    Túi-từ không thấy chữ "KHÔNG". Cách chữa ở đây là chỉ so trên các token
    ĐẶC TRƯNG của câu bị cấm (loại đi những token cũng xuất hiện trong câu bắt
    buộc), nên phần còn lại là {cay, kiểu, Xuyên} - không khớp gì nữa.

    Bài học cho hệ thật: đừng kiểm "vi phạm" bằng so khớp chuỗi. Hãy hỏi một VLM
    một câu hỏi nhị phân dứt khoát ("nước dùng trong hay đục?") và dùng câu trả
    lời đó. So khớp chuỗi không phân biệt được khẳng định với phủ định.
    """
    req = tokens(required) - set(exclude)
    if not req:
        return False
    for obs in observed:
        got = tokens(obs)
        if got and len(req & got) / len(req) >= 0.5:
            return True
    return False


@dataclass
class Entity:
    id: str
    name_vi: str
    name_en: str
    aliases: list[str]
    category: str
    region: str
    #: Ước lượng mức độ một T2I phổ thông vẽ đúng thực thể này mà không cần trợ giúp.
    #: KHÔNG phải số đo thực nghiệm - xem disclaimer trong entities.json.
    prior_strength: float
    must_have: list[str]
    must_not: list[str]
    confusable_with: list[dict[str, str]]
    wiki_title_vi: str | None = None
    notes: str | None = None
    #: Nhãn CLIP tiếng Anh mô tả (xem docs/ARCHITECTURE.md, CLIP cần nhãn mô tả, không phải tên trần).
    clip_label: str = ""
    #: v1.8 (Culture-TRIP "comparable objects"): một cụm so sánh với vật quen thuộc mà model T2I đã biết,
    #: vd "a long split tunic worn over wide trousers", "a giant round woven basket used as a boat". Vào prompt ngay sau tên.
    analogy_en: str = ""
    #: "object" | "context" - context (sự kiện, cảnh) không probe được bằng CLIP danh tính.
    kind: str = "object"
    #: Bản tiếng Anh viết tay của must_have / must_not, cùng thứ tự. Dùng thẳng cho prompt SDXL,
    #: không phụ thuộc VLM dịch (v1.1: dịch trả rỗng hoặc nhiễm "wide obi").
    must_have_en: list[str] = field(default_factory=list)
    must_not_en: list[str] = field(default_factory=list)
    #: v1.4.1: thẻ NGẮN cho prompt họ SDXL (thẻ phân biệt đứng đầu) và negative KHÔNG dùng danh từ của must_have
    #: (CLIP không hiểu phủ định: "dress with no trousers" trong negative đẩy ảnh xa "trousers").
    tags_en: list[str] = field(default_factory=list)
    neg_tags_en: list[str] = field(default_factory=list)

    @property
    def search_terms(self) -> list[str]:
        return [self.name_vi, self.name_en, *self.aliases]

    @property
    def primary_confusable(self) -> dict[str, str] | None:
        return self.confusable_with[0] if self.confusable_with else None


@dataclass
class KnowledgeBase:
    entities: dict[str, Entity] = field(default_factory=dict)
    version: str = ""
    disclaimer: str = ""

    @classmethod
    def load(cls, path: Path) -> "KnowledgeBase":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        kb = cls(version=raw.get("version", ""), disclaimer=raw.get("disclaimer", ""))
        for row in raw["entities"]:
            kb.entities[row["id"]] = Entity(**row)
        return kb

    def get(self, entity_id: str) -> Entity | None:
        return self.entities.get(entity_id)

    def add_adhoc(self, name_vi: str, name_en: str, category: str = "other",
                  region: str = "toan_quoc") -> Entity:
        """Đăng ký thực thể agent đề xuất, chưa có bằng chứng. Bằng chứng sẽ do stage extraction dựng."""
        slug = re.sub(r"[^a-z0-9]+", "_", normalize(name_vi)).strip("_")[:40]
        eid = f"x_{slug}"
        if eid in self.entities:
            return self.entities[eid]
        ent = Entity(
            id=eid, name_vi=name_vi, name_en=name_en, aliases=[name_vi, name_en],
            category=category, region=region,
            prior_strength=0.15,  # chưa biết -> coi là thấp, để vòng review sẵn sàng can thiệp mạnh
            must_have=[], must_not=[], confusable_with=[], wiki_title_vi=None,
            notes="thực thể do agent đề xuất lúc chạy, chưa có trong KB gốc",
            clip_label=f"a photo of Vietnamese {name_en}", kind="object",
        )
        self.entities[eid] = ent
        return ent

    def all(self) -> list[Entity]:
        return list(self.entities.values())

    # ------------------------------------------------------------------
    # Ghép text -> entity
    # ------------------------------------------------------------------

    def match_text(self, text: str, min_score: float = 0.0) -> list[tuple[Entity, float, str]]:
        """Tìm entity được nhắc tới trong text.

        Trả về (entity, score, alias đã khớp), sắp giảm dần theo score.
        Alias dài khớp được thì đáng tin hơn alias ngắn, nên score theo độ dài.
        """
        hits: list[tuple[Entity, float, str]] = []

        for ent in self.entities.values():
            best_score, best_term = 0.0, ""
            for term in ent.search_terms:
                if len(term) < 3 or not contains(text, term):
                    continue
                # Alias nhiều từ và dài thì đặc trưng hơn -> điểm cao hơn.
                score = min(1.0, 0.45 + 0.06 * len(term.split()) + 0.02 * len(term))
                if score > best_score:
                    best_score, best_term = score, term
            if best_score > min_score:
                hits.append((ent, best_score, best_term))

        hits.sort(key=lambda h: -h[1])
        return hits

    def match_terms(self, terms: list[str], min_score: float = 0.0) -> list[tuple[Entity, float, str]]:
        """Như match_text nhưng nhận sẵn danh sách keyword."""
        return self.match_text(" ; ".join(terms), min_score=min_score)

    def by_region(self, region: str) -> list[Entity]:
        return [e for e in self.entities.values() if e.region == region]

    def by_category(self, category: str) -> list[Entity]:
        return [e for e in self.entities.values() if e.category == category]
