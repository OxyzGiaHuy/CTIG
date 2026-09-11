"""
Từ điển và luật mở rộng keyword cho RuleAgent (chạy offline, không cần model).

Toàn bộ khối này là thứ PromptAgent thay bằng một lần gọi model. Giữ lại để:
  * test pipeline không cần GPU,
  * có baseline so với LLM: LLM suy ra được gì mà luật không suy ra được.
"""

from __future__ import annotations

from dataclasses import dataclass, field


#: Chỉ dấu vùng miền. Viết CÓ DẤU vì việc khớp dùng kb.contains() trên text còn
#: dấu (xem lý do trong docstring của contains). Các biến thể không dấu như
#: "hanoi", "sapa" được liệt riêng để hỗ trợ người nhập không dấu.
REGION_LEXICON: dict[str, list[str]] = {
    "bac_bo": [
        "hà nội", "hanoi", "bắc bộ", "bắc ninh", "hải phòng", "long biên",
        "đồng bằng sông hồng", "miền bắc", "kinh bắc",
    ],
    "trung_bo": [
        "huế", "hue", "hội an", "đà nẵng", "miền trung", "quảng nam",
        "nha trang", "quảng trị", "thừa thiên",
    ],
    "nam_bo": [
        "sài gòn", "saigon", "hồ chí minh", "miền tây", "miền nam", "cần thơ",
        "nam bộ", "cái răng", "cái bè", "đồng khởi", "mekong", "mê kông",
        "đồng bằng sông cửu long", "tiền giang",
    ],
    "tay_bac": [
        "mù cang chải", "sa pa", "sapa", "hoàng su phì", "yên bái", "lào cai",
        "hà giang", "tây bắc", "vùng cao", "miền núi phía bắc",
    ],
    "tay_nguyen": [
        "tây nguyên", "gia lai", "kon tum", "đắk lắk", "đắc lắc",
        "buôn ma thuột", "pleiku", "buôn làng",
    ],
}

#: Từ chỉ bối cảnh - không phải thực thể văn hoá nhưng cần cho việc dựng ảnh.
SCENE_LEXICON: dict[str, str] = {
    "buổi sáng": "buổi sáng",
    "sáng sớm": "sáng sớm",
    "bình minh": "bình minh",
    "trời chưa sáng": "trước lúc rạng sáng",
    "ban đêm": "ban đêm",
    "về đêm": "ban đêm",
    "đêm": "ban đêm",
    "buổi chiều": "buổi chiều",
    "mùa thu": "mùa thu",
    "mùa nước đổ": "mùa nước đổ",
    "sương mờ": "sương mù",
    "khói": "có khói",
    "phố cổ": "phố cổ",
    "sân khấu": "trên sân khấu",
    "bàn thờ": "trước bàn thờ",
    "bãi biển": "bãi biển",
    "bãi cát": "bãi cát",
    "hồ nước": "mặt hồ",
    "ruộng": "đồng ruộng",
    "chợ": "khu chợ",
    "quán": "hàng quán",
    "xưởng": "xưởng thủ công",
    "sân đình": "sân đình",
    "thuyền": "trên thuyền",
    "sông": "trên sông",
    "núi": "vùng núi",
    "đồi": "sườn đồi",
    "cổng trường": "cổng trường",
}


@dataclass
class ExpansionRule:
    """Một luật sinh "Keywords mới" - mũi tên thứ hai trong ô Analysis của sơ đồ."""

    add: list[str]
    why: str
    #: Kích hoạt khi entity này đã được nhận ra ở bước surface.
    when_entity: str | None = None
    #: Kích hoạt khi prompt chứa một trong các cụm này (khớp theo biên từ, còn dấu).
    when_text: list[str] = field(default_factory=list)
    #: Chỉ kích hoạt nếu vùng miền suy ra được nằm trong danh sách này (None = mọi vùng).
    only_region: list[str] | None = None
    #: Độ tin cậy của suy luận - dùng để xếp hạng ở bước Rank.
    confidence: float = 0.7


EXPANSION_RULES: list[ExpansionRule] = [
    # --- Đồng xuất hiện: thấy A thì gần như chắc có B ---
    ExpansionRule(when_entity="ganh_hang_rong", add=["non_la"], confidence=0.85,
                  why="người gánh hàng rong hầu như luôn đội nón lá"),
    ExpansionRule(when_entity="cho_noi", add=["non_la", "ao_ba_ba"], confidence=0.8,
                  why="người buôn ở chợ nổi miền Tây thường mặc áo bà ba và đội nón lá"),
    ExpansionRule(when_entity="quan_ho", add=["ao_tu_than", "non_quai_thao", "ao_the_khan_xep"],
                  confidence=0.9,
                  why="hát quan họ có trang phục quy chuẩn: liền chị áo tứ thân + nón quai thao, liền anh áo the khăn xếp"),
    ExpansionRule(when_entity="ao_tu_than", add=["non_quai_thao"], confidence=0.75,
                  why="áo tứ thân đi cặp với nón quai thao, KHÔNG phải nón lá chóp nhọn"),
    ExpansionRule(when_entity="non_quai_thao", add=["ao_tu_than"], confidence=0.75,
                  why="nón quai thao là phụ kiện của bộ áo tứ thân"),
    ExpansionRule(when_entity="ao_ba_ba", add=["khan_ran"], confidence=0.7,
                  why="áo bà ba miền Tây thường đi cùng khăn rằn"),
    ExpansionRule(when_entity="khan_ran", add=["ao_ba_ba"], confidence=0.7,
                  why="khăn rằn là phụ kiện của bộ áo bà ba"),
    ExpansionRule(when_entity="cong_chieng_tay_nguyen", add=["nha_rong"], confidence=0.7,
                  why="cồng chiêng thường diễn trước nhà rông của buôn làng"),
    ExpansionRule(when_entity="trung_thu", add=["mua_lan"], confidence=0.65,
                  why="Trung Thu ở Việt Nam thường có múa lân"),
    ExpansionRule(when_entity="thuyen_thung", add=["non_la"], confidence=0.6,
                  why="ngư dân miền Trung đi biển thường đội nón lá"),
    ExpansionRule(when_entity="nha_nhac_hue", add=["ao_the_khan_xep"], confidence=0.8,
                  why="nhạc công nhã nhạc mặc bào phục và khăn xếp"),
    ExpansionRule(when_entity="dan_bau", add=["ao_dai"], confidence=0.5,
                  why="nghệ sĩ nhạc cụ truyền thống biểu diễn thường mặc áo dài (suy luận yếu, dễ dư thừa)"),
    ExpansionRule(when_entity="dan_tranh", add=["ao_dai"], confidence=0.5,
                  why="nghệ sĩ đàn tranh biểu diễn thường mặc áo dài (suy luận yếu)"),
    ExpansionRule(when_entity="ruong_bac_thang", add=["trang_phuc_hmong"], confidence=0.55,
                  only_region=["tay_bac"],
                  why="ruộng bậc thang Tây Bắc do người H'Mông và Dao canh tác"),

    # --- Điều kiện hoá theo vùng: cùng dịp lễ nhưng vật thể khác nhau ---
    ExpansionRule(when_entity="tet_nguyen_dan", add=["banh_chung"], only_region=["bac_bo"],
                  confidence=0.85,
                  why="Tết miền Bắc dùng bánh chưng vuông và cành đào"),
    ExpansionRule(when_entity="tet_nguyen_dan", add=["banh_tet"], only_region=["nam_bo", "trung_bo"],
                  confidence=0.85,
                  why="Tết miền Trung và Nam dùng bánh tét hình trụ và mai vàng"),

    # --- Suy luận từ bối cảnh: prompt KHÔNG nêu tên thực thể ---
    ExpansionRule(when_text=["thầy đồ", "chữ nho", "văn tế", "đọc văn", "lớp học chữ"],
                  add=["ao_the_khan_xep", "dinh_lang"], confidence=0.7,
                  why="cảnh nho học cổ: người đọc mặc áo the khăn xếp, không gian thường là đình hoặc nhà thờ họ"),
    ExpansionRule(when_text=["ăn hỏi", "cô dâu", "chú rể", "đám cưới", "lễ cưới"],
                  add=["ao_dai", "ao_the_khan_xep"], confidence=0.75,
                  why="lễ cưới truyền thống Việt: cô dâu áo dài, chú rể áo dài nam hoặc áo the"),
    ExpansionRule(when_text=["khắc bản gỗ", "bản gỗ", "in tranh", "tranh dân gian", "khắc gỗ"],
                  add=["tranh_dong_ho"], confidence=0.7,
                  why="nghề khắc bản gỗ in tranh dân gian ở Việt Nam gắn với tranh Đông Hồ"),
    ExpansionRule(when_text=["ghế nhựa thấp", "quán ăn ven đường", "quán cóc", "hàng ăn ven đường"],
                  add=["pho", "ca_phe_sua_da"], confidence=0.6,
                  why="quán ven đường ghế nhựa thấp: món phổ biến nhất là phở và cà phê sữa đá"),
    ExpansionRule(when_text=["bán hoa", "người bán", "đạp xe chở", "chở đầy hoa", "hàng rong"],
                  add=["non_la", "ganh_hang_rong"], confidence=0.6,
                  why="người bán hàng lưu động thường đội nón lá, dùng gánh hoặc xe đạp thồ"),
    ExpansionRule(when_text=["miền tây", "dừa", "xoài", "ra chợ", "sông nước"],
                  add=["cho_noi", "ao_ba_ba", "non_la"], confidence=0.7,
                  only_region=["nam_bo"],
                  why="cảnh thuyền chở trái cây ra chợ sớm ở miền Tây là chợ nổi"),
    ExpansionRule(when_text=["mâm ngũ quả", "bàn thờ gia tiên", "bao lì xì", "cây quất",
                             "cành đào", "cây mai", "mâm cỗ tết", "chợ tết",
                             "năm mới âm lịch", "mừng năm mới"],
                  add=["tet_nguyen_dan"], confidence=0.85,
                  why="các vật thể này chỉ xuất hiện trong Tết Nguyên Đán"),
    ExpansionRule(when_text=["rước đèn", "đèn ông sao", "bánh nướng", "bánh dẻo"],
                  add=["trung_thu"], confidence=0.85,
                  why="rước đèn ông sao và bánh nướng bánh dẻo là dấu hiệu Trung Thu"),
    ExpansionRule(when_text=["đầu lân", "ông địa", "cột hoa mai"],
                  add=["mua_lan"], confidence=0.8,
                  why="đầu lân và ông Địa là dấu hiệu múa lân"),
    ExpansionRule(when_text=["nữ sinh", "học sinh", "đồng phục", "cổng trường", "trung học"],
                  add=["ao_dai"], confidence=0.7,
                  why="đồng phục nữ sinh trung học Việt Nam nhiều nơi là áo dài trắng"),
    ExpansionRule(when_text=["hoa tam giác mạch", "khèn", "phiên chợ vùng cao"],
                  add=["trang_phuc_hmong"], confidence=0.65,
                  why="hoa tam giác mạch và khèn gắn với vùng người H'Mông"),
    ExpansionRule(when_text=["làng chài", "ngư dân"], add=["thuyen_thung"], confidence=0.5,
                  only_region=["trung_bo"],
                  why="làng chài miền Trung đặc trưng bởi thuyền thúng tròn"),
]
