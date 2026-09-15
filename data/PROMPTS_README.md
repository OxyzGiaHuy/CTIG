Vietnamese Cultural Prompts

**Bộ prompt song ngữ Việt–Anh để đánh giá năng lực sinh ảnh văn hoá Việt Nam của mô hình text-to-image.**

## Cấu trúc

```
data/
├── prompts_simple.json    # 50 prompt, mỗi prompt 1 entity
└── prompts_complex.json   # 50 prompt, mỗi prompt ≥2 entity
```

Cả hai file dùng chung schema:

```json
{
  "id": "C046",
  "text_vi": "Người phụ nữ Tày mặc áo chàm ngồi hát then, tay gảy cây đàn tính, phía sau là vách gỗ của ngôi nhà sàn.",
  "text_en": "A Tay woman in an indigo tunic sits singing then melodies, plucking a tinh lute, the wooden wall of a stilt house behind her.",
  "categories": ["Arts & Music", "Clothing", "Architecture & Landmark"],
  "entities": ["hát then", "đàn tính", "áo chàm người Tày", "nhà sàn"]
}
```

| Trường | Mô tả |
|---|---|
| `id` | `S001`–`S050` (simple) · `C001`–`C050` (complex) |
| `text_vi` / `text_en` | Hai bản tương đương về nội dung, không bản nào thừa hay thiếu thông tin |
| `categories` | Category của prompt, lấy từ bộ 8 category bên dưới |
| `entities` | Các entity văn hoá **được nêu trong prompt** |

## Hai tập

|  | `prompts_simple` | `prompts_complex` |
|---|---|---|
| Số prompt | 50 | 50 |
| Entity / prompt | **đúng 1** | **≥ 2** (trung vị 3, tối đa 7) |
| Category / prompt | đúng 1 | 1–3 (đa số 2) |
| Độ dài `text_vi` | 7–19 từ (trung vị 12) | 15–41 từ (trung vị 24) |
| Số entity duy nhất | 50 (không trùng lặp) | 141 |

**Tập simple** kiểm tra khả năng dựng đúng **một** khái niệm văn hoá. Mỗi prompt nhắm vào một entity riêng biệt, không prompt nào trùng entity với prompt khác.

**Tập complex** kiểm tra khả năng **bố cục nhiều khái niệm** cùng lúc, có ràng buộc về vị trí, thời điểm hoặc hoạt động giữa chúng. Tập này kế thừa từ tập simple: cùng chủ thể văn hoá nhưng thêm bối cảnh, thêm entity tương tác, hoặc thêm quy trình đang diễn ra.

## Thống kê category

| Category | simple | complex | Tổng lượt |
|---|---:|---:|---:|
| Clothing | 8 | 18 | 26 |
| Everyday Life & Trades | 6 | 20 | 26 |
| Arts & Music | 8 | 14 | 22 |
| Food & Drink | 9 | 11 | 20 |
| Architecture & Landmark | 6 | 13 | 19 |
| Landscape | 4 | 10 | 14 |
| Customs & Rituals | 5 | 7 | 12 |
| Festival | 4 | 6 | 10 |
| **Tổng** | **50** | **99** | **149** |

Tập simple mỗi prompt một category nên tổng bằng 50. Tập complex cho phép nhiều category nên tổng lượt gán (99) lớn hơn số prompt.

**Ý nghĩa 8 category:** *Clothing* trang phục · *Food & Drink* ẩm thực · *Arts & Music* nghệ thuật biểu diễn và mỹ thuật dân gian · *Everyday Life & Trades* sinh hoạt và nghề thủ công · *Architecture & Landmark* kiến trúc và di tích · *Landscape* cảnh quan · *Customs & Rituals* phong tục và nghi lễ · *Festival* lễ hội.

## Nguyên tắc thiết kế

Ba quy ước được áp dụng nhất quán cho cả 100 prompt.

**1. Prompt mô tả một giá trị hoặc thực hành văn hoá biểu diễn được bằng hình ảnh.** Mọi thứ prompt nêu ra đều phải nhìn thấy được trong ảnh và người xem xác nhận được đúng hay sai.

**2. Prompt nêu tên thực tế, hạn chế mô tả đặc điểm.** Ví dụ: Prompt viết *"đàn tranh"* chứ không viết *"đàn tranh mười sáu dây"*. Chi tiết nhận dạng nằm ngoài prompt, để mô hình hoặc phương pháp bổ sung tri thức tự tìm. Nếu viết sẵn vào prompt thì mô hình nền cũng sinh đúng và khoảng cách giữa các phương pháp bị triệt tiêu.

**3. Chỉ giữ tên riêng biểu diễn được bằng hình ảnh.** Các địa danh có hình dáng dễ nhận diện được thì giữ (VD: Văn Miếu, Chùa Cầu Hội An...). Các tên hành chính không có dấu hiệu thị giác riêng thì bỏ, thay bằng mô tả chung (VD: *chợ nổi Cái Răng* → *chợ nổi*, *làng hương Quảng Phú Cầu* → *làng hương*...).

## Phạm vi và hạn chế

- **Độ phủ chưa đồng đều.** Category *Festival* và *Customs & Rituals* mỏng hơn các category khác. Về vùng miền, văn hoá Bắc Bộ chiếm tỷ trọng cao hơn Trung và Nam Bộ. Về dân tộc, dataset có H'Mông, Dao Đỏ, Thái, Tày, Gia Rai, Chăm — còn nhiều nhóm chưa được đại diện.
- **Một số prompt vẫn mô tả chi tiết ở mức nhẹ**, chủ yếu ở category *Food & Drink*, được giữ lại có cân nhắc để câu văn tự nhiên.
