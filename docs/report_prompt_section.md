# Phần báo cáo: Bộ prompt và ảnh tham chiếu

*(Soạn từ thư mục `Data/` trên Drive SOICT'26, ngày 12/09/2026: `prompts_vi.jsonl`, `prompts_complex.json`,
`ref_images.zip`, `ref_images_complex.zip`. Số liệu tính trực tiếp trên file.)*

## 1. Mục tiêu của bộ prompt

Bộ prompt là thước đo của toàn đề tài: mỗi prompt mô tả một cảnh mang thực thể văn hoá Việt Nam mà mô hình
text-to-image hay vẽ sai hoặc thay bằng thực thể của văn hoá lân cận (áo dài thành qipao/kimono, bánh chưng thành
zongzi, Tết thành lồng đèn Trung Quốc). Prompt được viết bằng tiếng Việt có bản dịch tiếng Anh, kèm nhãn vàng là
danh sách thực thể phải xuất hiện, để (a) khối Analysis có đáp án khi tách keywords, (b) khối Search có đơn vị để
truy hồi bằng chứng, (c) khối Đánh giá có căn cứ chấm "đúng hay sai văn hoá" thay vì chỉ "đẹp hay xấu".

Hiện có hai bộ, xây theo hai giai đoạn.

## 2. Bộ 1: 50 prompt cơ bản (`prompts_vi.jsonl`)

Mỗi prompt xoay quanh **một** thực thể chính (35/50 prompt có đúng một thực thể vàng), câu ngắn (trung bình 14 từ),
đủ để cô lập một lỗi văn hoá. Trường dữ liệu: `id`, `text_vi`, `text_en`, `gold_entities` (mã thực thể trong cơ sở
tri thức 38 thực thể của pipeline), `category`, `difficulty`, `note`.

| | số prompt |
|---|---|
| độ khó: dễ / trung bình / khó | 6 / 17 / 27 |
| nhóm: lễ hội / ẩm thực / sinh hoạt / trang phục / kiến trúc / nghệ thuật / nhạc cụ / cảnh quan | 10 / 9 / 8 / 7 / 5 / 5 / 4 / 2 |
| thực thể vàng mỗi prompt: 1 / 2 / 3 / 4 | 35 / 11 / 2 / 1 |

Độ khó gán theo mức "prior" của mô hình sinh: áo dài, phở là dễ vì mô hình đã thấy nhiều; thuyền thúng, áo tứ thân,
đàn bầu là khó vì mô hình gần như không biết và sẽ thay bằng thứ gần nhất nó biết. Bộ này đang được dùng để chạy
pipeline trên Kaggle (p001, p012, p031, p050).

## 3. Bộ 2: 45 prompt phức hợp (`prompts_complex.json`)

Đây là bước chuẩn bị cho bộ prompt cuối. Khác bộ 1 ở ba điểm:

1. **Nhiều thực thể trong một cảnh.** 118 lượt thực thể trên 45 prompt, 117 thực thể khác nhau; 24/45 prompt có
   từ 3 thực thể trở lên (3: 17 prompt, 4: 5, 5: 2). Ví dụ C008 ghép áo the, khăn xếp, áo tứ thân, nón quai thao,
   quan họ và thuyền rồng trong cùng một cảnh hội Lim.
2. **Câu dài và có bố cục.** Trung bình 25 từ (14 đến 42), mô tả vị trí tương đối, thời điểm, hành động: *"Ông đồ ngồi
   trên chiếu viết thư pháp bằng mực tàu lên giấy đỏ, phía sau là cổng Văn Miếu - Quốc Tử Giám."*
3. **Phân loại theo tám nhóm văn hoá bằng tiếng Anh**, mỗi prompt có thể thuộc nhiều nhóm (27/45 thuộc 2 nhóm, 6 thuộc
   3 nhóm), phù hợp để đối chiếu với các benchmark văn hoá quốc tế (CultDiff, CuRe, CULTIVate):

| nhóm | số prompt | nhóm | số prompt |
|---|---|---|---|
| Everyday Life & Trades | 18 | Landscape | 9 |
| Clothing | 13 | Food & Drink | 8 |
| Performing Arts & Music | 13 | Festival | 7 |
| Architecture & Landmark | 11 | Customs & Rituals | 5 |

Độ khó: dễ 17, trung bình 19, khó 9. Thực thể được ghi bằng tên tiếng Việt tự nhiên (`"đàn nguyệt"`, `"mâm ngũ quả"`),
chưa gắn mã trong cơ sở tri thức.

**Vì sao cần bộ 2.** Bộ 1 đo lỗi *thay thế thực thể*. Bộ 2 đo thêm hai lỗi chỉ xuất hiện khi cảnh phức tạp: lỗi
*bỏ sót* (mô hình vẽ 2 trong 4 thực thể) và lỗi *trộn văn hoá* (vẽ đúng áo tứ thân nhưng thuyền rồng thành thuyền
Trung Quốc). Đây cũng là dạng prompt người dùng thật hay viết.

## 4. Ảnh tham chiếu từ API search

Với mỗi prompt, bạn cùng nhóm đã dùng API tìm ảnh (truy vấn bằng `text_vi`, lùi về `text_en` khi cần) và tải về:

| | bộ 1 | bộ 2 |
|---|---|---|
| ảnh mỗi prompt | 10 | 20 |
| tổng ảnh | 500 (50 prompt) | 900 (45 prompt) |
| truy vấn thành công | 50/50 (3 prompt phải lùi về tiếng Anh) | 45/45 |
| định dạng | jpg 451, webp 29, png 20 | jpg 815, png 50, webp 34, gif 1 |
| kích cỡ file trung vị | 191 KB | 236 KB |

Kèm theo có `download_log.jsonl` (truy vấn dùng, số ảnh, lỗi) và `gallery.html` để duyệt mắt. Ảnh có độ phân giải
tốt (mẫu kiểm: 700 đến 2048 px cạnh dài), gần như không có file hỏng.

**Nhận xét chất lượng (kiểm bằng mắt trên mẫu p001, C001, C008):**

- Ảnh khớp **thực thể** tốt: ảnh C008 là hội Lim thật, có áo tứ thân, nón quai thao, thuyền trên hồ. Đây đúng là thứ
  khối Generation cần làm ảnh tham chiếu (IP-Adapter) và khối Đánh giá cần làm mốc so sánh.
- Ảnh khớp **cảnh** kém hơn: ảnh ông đồ C001 là người trẻ ngồi bàn, không có chiếu, không có cổng Văn Miếu. Hợp lý,
  vì search trả ảnh theo từ khoá nổi bật nhất, không theo cả câu. Vì thế ảnh nên được gắn theo **thực thể** (một thư mục
  cho "ông đồ", một cho "Văn Miếu") thay vì theo prompt; pipeline hiện cũng làm việc này bằng CLIP để lọc ảnh nào
  thuộc thực thể nào.
- Có ảnh đã qua xử lý AI: p001 có 4 ảnh tên `dreamina-...` (ảnh thật được phóng nét bằng công cụ AI, còn watermark
  trang bán hàng). Ảnh tham chiếu cho nghiên cứu văn hoá nên loại ảnh có dấu AI để tránh "tự tham chiếu" lỗi của mô hình.
- Chưa có thông tin nguồn và giấy phép cho từng ảnh. Để dùng trong bài báo hay user study cần ít nhất URL gốc; về lâu
  dài ưu tiên nguồn có giấy phép rõ (Wikimedia Commons, Openverse).
- Nhiều ảnh là ảnh nhóm hoặc ảnh sự kiện. Lần chạy v1.3 cho thấy IP-Adapter kéo cả bố cục ảnh tham chiếu (ảnh nhóm
  nữ sinh làm ảnh sinh có 3 đến 4 người dù prompt là "một cô gái"), nên khi chọn ảnh tham chiếu cần thêm tiêu chí
  khớp bố cục prompt, không chỉ khớp thực thể.

## 5. Liên hệ với pipeline và việc còn lại

- **Độ phủ cơ sở tri thức.** Chỉ 47/118 lượt thực thể của bộ 2 có trong cơ sở tri thức 38 thực thể viết tay hiện nay
  (khớp tên hoặc tên gọi khác). 70 thực thể còn lại (đàn nguyệt, trống bản, Điện Thái Hoà, mâm ngũ quả, lễ hội Katê,
  tháp Po Klong Garai, giấy điệp, ...) là lý do khối Search phải **tự rút** must_have / must_not từ văn bản thay vì
  dựa vào tri thức viết tay. Đây là đúng hướng của bản draft ban đầu: Search tạo ra bằng chứng, không chỉ xác nhận.
- **Chuẩn hoá nhãn.** Bộ 2 cần một bước gắn mã thực thể (`ong_do`, `van_mieu`, ...) và tách "thực thể" khỏi "bối cảnh"
  (kind object / context) để khối Đánh giá chấm được từng thực thể; có thể làm bán tự động bằng chính Analysis agent
  rồi người kiểm.
- **Bộ prompt cuối** dự kiến: giữ cấu trúc bộ 2, cân lại độ khó (bộ 2 hơi thiên dễ: 17 dễ / 9 khó, ngược với bộ 1),
  bổ sung nhóm còn mỏng (Customs & Rituals 5, Landscape 9), mỗi prompt kèm mã thực thể, thuộc tính thị giác kiểm được,
  và 3 đến 5 ảnh tham chiếu đã lọc có nguồn.
