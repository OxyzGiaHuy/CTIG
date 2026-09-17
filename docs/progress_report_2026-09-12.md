# CTIG - Cultural Text-to-Image cho miền Việt Nam: báo cáo tiến độ

*Nhóm SOICT'26 · 12/09/2026 · mã nguồn: github.com/OxyzGiaHuy/CTIG (v1.3.1) · dữ liệu: Drive SOICT'26/Data*

## 1. Bài toán và kiến trúc

Mô hình text-to-image (SDXL và họ hàng) khi nhận prompt về văn hoá Việt Nam hay **thay thực thể** bằng thực thể của
văn hoá đông dữ liệu hơn: áo dài thành qipao hoặc váy liền xẻ tà không quần, bánh chưng thành zongzi, Tết thành lồng
đèn Trung Quốc. Đề tài xây một pipeline agentic bám theo bản draft ban đầu:

```
Prompt (VI) ─► [1] Analysis agent: keywords, thực thể ─► [2] Search: Wikipedia · web · ảnh (API, tiếng Việt)
            ─► [2b] Rút bằng chứng: must_have / must_not / dễ nhầm với ─► [3] Hợp đồng văn hoá → prompt sinh
            ─► [4] Sinh ảnh nhiều model (SDXL, RealVisXL, LoRA áo dài, IP-Adapter ảnh tham chiếu)
            ─► [5] Chấm: CLIP thuộc tính · BLIP-2 ITM · PickScore ─► (Review loop VLM: cờ, đang tắt để tối ưu ảnh trước)
```

Mọi bước ghi đầu ra ra đĩa và có ba lớp cache (gọi LLM, web, từng bước) nên chạy lại không tốn API; notebook Kaggle
hiển thị đầu ra từng bước để kiểm bằng mắt.

## 2. Trạng thái từng khối

| khối | trạng thái | ghi chú |
|---|---|---|
| Analysis (Qwen2.5-VL-3B) | chạy, có cache | tách keywords, dịch EN, giới hạn 6 thực thể ứng viên có căn cứ |
| Search (Wikipedia VI, DuckDuckGo VI/EN, Commons) | chạy, có cache | truy vấn theo thực thể cho văn bản, theo prompt gốc cho ảnh; tải toàn văn 3 trang đầu |
| Rút bằng chứng (VLM đọc văn bản) | chạy | mỗi thuộc tính kèm câu gốc; lọc rác; thuộc tính trùng tri thức viết tay tính là "xác nhận" |
| Hợp đồng văn hoá → prompt | chạy | must_have EN vào prompt, must_not EN vào negative (từ v1.2.1) |
| Sinh ảnh nhiều model | chạy | 7 model, 4 ứng viên/model, DPM++ 2M Karras, compel cho prompt dài, hires ×1,5 |
| Chấm điểm | chạy | CLIP thuộc tính là thước đo tách được lỗi; PickScore đo thẩm mỹ (mới nạp được ở lần chạy 12/09) |
| Review loop (VLM phê bình, sửa prompt) | tắt | VLM 3B trả "có" cho mọi câu hỏi đóng; cần thiết kế lại sau khi ảnh nền đã tốt |
| Đánh giá người dùng (user study) | chưa | đã có mẫu `user_study.csv` xuất từ pipeline |

## 3. Kết quả thực nghiệm trên p001 qua các phiên bản

Prompt p001: *"Một cô gái mặc áo dài trắng đứng trước cổng trường."* Cùng seed 1234, cùng hợp đồng văn hoá, chỉ đổi
cách sinh và bộ model. Thước đo chính là **CLIP attr**: phần xác suất CLIP dành cho câu "áo dài với &lt;must_have&gt;"
(quần dài ống rộng, cổ đứng, hai tà xẻ hông) so với "với &lt;must_not&gt;" (váy liền không quần, obi, cổ chéo). Danh tính
CLIP và BLIP-2 ITM bão hoà 0,95 đến 1,00 ở mọi ảnh nên không dùng để xếp hạng.

### 3.1. v1.2.1 (11/09): 6 model × 2 ứng viên, negative chỉ có tên kimono/qipao/hanbok

| model | attr TB | attr tốt nhất | ITM attr TB | giây | VRAM |
|---|---|---|---|---|---|
| sdxl_turbo | 0,43 | 0,45 | 0,78 | 51 | 7,1 GB |
| dreamshaper8 (SD1.5) | 0,42 | 0,67 | 0,85 | 22 | 2,6 GB |
| sdxl_base | 0,42 | 0,45 | 0,41 | 69 | 7,1 GB |
| sdxl_aodai (LoRA) | 0,46 | 0,52 | 0,74 | 84 | 7,2 GB |
| sdxl_ref (IP-Adapter) | 0,42 | 0,51 | 0,60 | 94 | 11,2 GB |
| playground25 | 0,16 | 0,18 | 0,79 | 91 | 7,1 GB |

Nhìn grid: 4/12 ảnh là váy liền xẻ tà **không quần** ("qipao hoá"), 1 ảnh có đai đỏ. CLIP attr xếp đúng các ảnh này
xuống dưới, trong khi danh tính vẫn cho 0,99. Cùng một model, hai seed lệch nhau nhiều hơn hai model khác nhau.

![Grid p001 v1.2.1](report_assets/v121_grid.jpg)

*Hình 1. p001 v1.2.1, 6 model × 2 ứng viên. Hàng 2 ảnh 2, hàng 3 ảnh 2, hàng 5 ảnh 2 và hàng 6: váy liền không quần.*

### 3.2. v1.3 (12/09): 7 model × 4 ứng viên, negative theo must_not, DPM++ Karras, hires ×1,5

| model | attr TB | attr tốt nhất | ITM attr TB | giây/4 ảnh | VRAM |
|---|---|---|---|---|---|
| dreamshaper8 (SD1.5) | 0,53 | 0,67 | 0,92 | 52 | 2,6 GB |
| sdxl_base | 0,56 | 0,61 | 0,86 | 230 | 13,4 GB |
| **realvis_xl** | **0,75** | 0,80 | 0,96 | 230 | 13,4 GB |
| sdxl_aodai (SDXL + LoRA) | 0,59 | 0,70 | 0,90 | 261 | 13,4 GB |
| **realvis_aodai** (RealVis + LoRA) | 0,60 | **0,86** | 0,96 | 255 | 13,4 GB |
| sdxl_refplus (IP-Adapter Plus, 3 ảnh) | 0,80 | 0,89 | 0,74 | 217 | 9,2 GB |
| playground25 | 0,16 | 0,21 | 0,87 | 239 | 7,2 GB |

![Grid p001 v1.3](report_assets/v13_grid.jpg)

*Hình 2. p001 v1.3, 7 model × 4 ứng viên (viền xanh: ứng viên CLIP chọn). Gần như tất cả 28 ảnh có quần dài.*

Ba kết luận từ lần chạy này:

1. **Negative theo must_not chặn được "qipao hoá".** Từ 8/12 ảnh có quần lên ~28/28. Tác dụng phụ: dreamshaper8 cho
   2 ảnh kiểu vest trắng, sdxl_base cho kiểu áo khoác dài; điểm thuộc tính vẫn xếp đúng các ca này thấp.
2. **RealVisXL hơn SDXL base trên cùng seed** (attr 0,75 so với 0,56); RealVisXL cộng LoRA áo dài cho ảnh tốt nhất
   toàn grid (0,86). Playground v2.5 bạc màu ở cả 4 ảnh vì lỗi thay VAE của chúng tôi, đã hoàn lại.
3. **IP-Adapter với ảnh tham chiếu nhóm kéo theo bố cục**: 3/4 ảnh có 3 đến 4 người dù prompt là "một cô gái"; điểm
   thuộc tính cao nhưng ITM attr thấp. Đã hạ trọng số và đổi cách chọn ảnh tham chiếu (mục 5.4).

![Hàng realvis_aodai](report_assets/v13_realvis_aodai_row.jpg)

*Hình 3. Hàng RealVisXL + LoRA áo dài: ứng viên 3 đạt attr 0,86, cao nhất 28 ảnh.*

## 4. Kết luận đến nay và giả thuyết đang kiểm

**Đã có bằng chứng (n nhỏ, cần thêm prompt):**
- Thước đo danh tính (CLIP, BLIP-2 ITM) bão hoà trên prompt dễ; **thước đo mức thuộc tính** (must_have với must_not)
  mới tách được ảnh đúng và ảnh sai văn hoá, khớp mắt người.
- Lỗi thay thế văn hoá của áo dài trong họ SDXL là "qipao hoá" (váy liền không quần), không phải kimono; negative
  theo tên confusable không chặn được, negative theo thuộc tính thì được.
- Search từ keywords tốt hơn cho **văn bản** (ra Wikipedia, bài cấu tạo), search từ prompt gốc tốt hơn cho **ảnh**.
- Phương sai theo seed lớn hơn phương sai giữa model ở n=2, nên so model cần n ≥ 4 (đã nâng lên 4).

**Đang kiểm:** best-of-4 chọn theo điểm tổng cho ảnh cuối đúng hơn ảnh seed đầu (H10); hires ×1,5 tăng PickScore
mà không giảm attr (H11); RealVisXL hơn SDXL base trên ≥ 4 prompt trang phục (H12); LoRA áo dài hơn model gốc (H8).

## 5. Bộ prompt và ảnh tham chiếu

### 1. Mục tiêu của bộ prompt

Bộ prompt là thước đo của toàn đề tài: mỗi prompt mô tả một cảnh mang thực thể văn hoá Việt Nam mà mô hình
text-to-image hay vẽ sai hoặc thay bằng thực thể của văn hoá lân cận (áo dài thành qipao hay kimono, bánh chưng thành
zongzi, Tết thành lồng đèn Trung Quốc). Prompt được viết bằng tiếng Việt có bản dịch tiếng Anh, kèm nhãn vàng là
danh sách thực thể phải xuất hiện, để (a) khối Analysis có đáp án khi tách keywords, (b) khối Search có đơn vị để
truy hồi bằng chứng, (c) khối Đánh giá có căn cứ chấm "đúng hay sai văn hoá" thay vì chỉ "đẹp hay xấu".

Hiện có hai bộ, xây theo hai giai đoạn.

### 2. Bộ 1: 50 prompt cơ bản (`prompts_vi.jsonl`)

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

### 3. Bộ 2: 45 prompt phức hợp (`prompts_complex.json`)

Đây là bước chuẩn bị cho bộ prompt cuối. Khác bộ 1 ở ba điểm:

1. **Nhiều thực thể trong một cảnh.** 118 lượt thực thể trên 45 prompt, 117 thực thể khác nhau; 24/45 prompt có
   từ 3 thực thể trở lên (3 thực thể: 17 prompt, 4: 5, 5: 2). Ví dụ C008 ghép áo the, khăn xếp, áo tứ thân, nón quai
   thao, quan họ và thuyền rồng trong cùng một cảnh hội Lim.
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

### 4. Ảnh tham chiếu từ API search

Với mỗi prompt, nhóm đã dùng API tìm ảnh (truy vấn bằng `text_vi`, lùi về `text_en` khi cần) và tải về:

| | bộ 1 | bộ 2 |
|---|---|---|
| ảnh mỗi prompt | 10 | 20 |
| tổng ảnh | 500 (50 prompt) | 900 (45 prompt) |
| truy vấn thành công | 50/50 (3 prompt phải lùi về tiếng Anh) | 45/45 |
| định dạng | jpg 451, webp 29, png 20 | jpg 815, png 50, webp 34, gif 1 |
| kích cỡ file trung vị | 191 KB | 236 KB |

Kèm theo có `download_log.jsonl` (truy vấn dùng, số ảnh, lỗi) và `gallery.html` để duyệt bằng mắt. Ảnh có độ phân
giải tốt (mẫu kiểm: 700 đến 2048 px cạnh dài), gần như không có file hỏng.

### 4.1. Ảnh khớp thực thể tốt

Prompt C008 (hội Lim) là ví dụ thuận lợi: 20 ảnh đều là hội Lim hoặc quan họ thật, có áo tứ thân, nón quai thao,
khăn mỏ quạ, áo the khăn xếp, thuyền rồng trên hồ. Đây đúng là thứ khối Generation cần làm ảnh tham chiếu (IP-Adapter)
và khối Đánh giá cần làm mốc so sánh.

![C008: 20 ảnh tham chiếu cho hội Lim](report_assets/C008_refs.jpg)

*Hình 1. 20 ảnh tham chiếu của C008. Các thực thể trong prompt đều xuất hiện, nhưng rải trên nhiều ảnh: không ảnh nào
có đủ cả cảnh.*

### 4.2. Ảnh khớp cảnh kém hơn

Search trả ảnh theo từ khoá nổi bật nhất trong câu, không theo cả câu. Với C001, 20 ảnh đều là "ông đồ viết thư
pháp", nhưng gần như không ảnh nào có cổng Văn Miếu, nhiều ảnh là người trẻ ngồi bàn thay vì ngồi chiếu.

![C001: 20 ảnh tham chiếu](report_assets/C001_refs.jpg)

*Hình 2. C001 "Ông đồ ngồi trên chiếu viết thư pháp bằng mực tàu lên giấy đỏ, phía sau là cổng Văn Miếu". Ảnh khớp
thực thể "ông đồ, thư pháp, giấy đỏ" nhưng không khớp bối cảnh "chiếu", "cổng Văn Miếu".*

Vì thế ảnh nên được gắn theo **thực thể** (một nhóm cho "ông đồ", một cho "Văn Miếu") thay vì theo prompt. Pipeline
hiện làm việc này tự động: với mỗi thực thể, CLIP chấm từng ảnh theo mô tả tiếng Anh của thực thể đó và chỉ giữ ảnh
vượt ngưỡng làm tham chiếu.

### 4.3. Ba điểm cần xử lý

**(a) Ảnh đã qua công cụ AI.** p001 có 4/10 ảnh tên `dreamina-...`: ảnh chụp thật được phóng nét bằng công cụ AI
Dreamina, còn watermark trang bán áo dài. Ảnh tham chiếu cho nghiên cứu văn hoá nên loại ảnh có dấu AI để tránh
"tự tham chiếu" lỗi của chính mô hình sinh.

![p001: 10 ảnh tham chiếu, viền đỏ là ảnh qua công cụ AI](report_assets/p001_refs.jpg)

*Hình 3. p001. Bốn ảnh viền đỏ đã qua công cụ AI (nhận ra từ tên file); nội dung vẫn là nữ sinh áo dài thật nhưng
không nên dùng làm mốc "sự thật".*

**(b) Chưa có nguồn và giấy phép từng ảnh.** `download_log.jsonl` ghi số ảnh nhưng không ghi URL gốc. Để dùng trong
bài báo hay user study cần ít nhất URL nguồn; về lâu dài ưu tiên nguồn có giấy phép rõ (Wikimedia Commons, Openverse).

**(c) Ảnh nhóm kéo theo bố cục.** Nhiều ảnh là ảnh nhóm hoặc ảnh sự kiện. Lần chạy pipeline v1.3 cho thấy IP-Adapter
kéo cả bố cục của ảnh tham chiếu: với p001 "một cô gái", ảnh tham chiếu là ảnh nhóm nữ sinh nên 3/4 ảnh sinh ra có
3 đến 4 người. Khi chọn ảnh tham chiếu cần thêm tiêu chí khớp bố cục prompt, không chỉ khớp thực thể.

![Hàng sdxl_refplus trong grid v1.3](report_assets/v13_refplus_row.jpg)

*Hình 4. Hàng `sdxl_refplus` (RealVisXL + IP-Adapter Plus với 3 ảnh tham chiếu) của p001 v1.3: trang phục đúng
(điểm thuộc tính 0,67 đến 0,89) nhưng số người sai. Đã sửa bằng cách hạ trọng số IP-Adapter và xếp ảnh tham chiếu
theo độ khớp với prompt.*

Để đối chiếu, hàng dùng LoRA áo dài trên cùng seed cho ảnh đúng cả trang phục lẫn bố cục:

![Hàng realvis_aodai trong grid v1.3](report_assets/v13_realvis_aodai_row.jpg)

*Hình 5. Hàng `realvis_aodai` (RealVisXL + LoRA áo dài), p001 v1.3. Ứng viên thứ ba đạt điểm thuộc tính 0,86, cao
nhất trong 28 ảnh của lần chạy.*

### 5. Liên hệ với pipeline và việc còn lại

- **Độ phủ cơ sở tri thức.** Chỉ 47/118 lượt thực thể của bộ 2 có trong cơ sở tri thức 38 thực thể viết tay hiện nay
  (khớp tên hoặc tên gọi khác). 70 thực thể còn lại (đàn nguyệt, trống bản, Điện Thái Hoà, mâm ngũ quả, lễ hội Katê,
  tháp Po Klong Garai, giấy điệp, ...) là lý do khối Search phải **tự rút** must_have / must_not từ văn bản thay vì
  dựa vào tri thức viết tay. Đây đúng hướng của bản draft ban đầu: Search tạo ra bằng chứng, không chỉ xác nhận.
- **Chuẩn hoá nhãn.** Bộ 2 cần một bước gắn mã thực thể (`ong_do`, `van_mieu`, ...) và tách "thực thể" khỏi "bối cảnh"
  (kind object / context) để khối Đánh giá chấm được từng thực thể; có thể làm bán tự động bằng chính Analysis agent
  rồi người kiểm.
- **Bộ prompt cuối** dự kiến: giữ cấu trúc bộ 2, cân lại độ khó (bộ 2 hơi thiên dễ: 17 dễ / 9 khó, ngược với bộ 1),
  bổ sung nhóm còn mỏng (Customs & Rituals 5, Landscape 9), mỗi prompt kèm mã thực thể, thuộc tính thị giác kiểm được,
  và 3 đến 5 ảnh tham chiếu đã lọc, có nguồn, không qua công cụ AI.


## 6. Kế hoạch

1. Chạy p001 lại với v1.3.1 (đang chạy; PickScore đã nạp), rồi p031 (áo tứ thân), p050 (Tết), p012 (thuyền thúng)
   để kiểm H10 đến H12 trên thực thể prior thấp, xuất báo cáo HTML ảnh gốc từ pipeline.
2. Gắn mã thực thể cho 45 prompt phức hợp bằng Analysis agent rồi người kiểm; mở rộng cơ sở tri thức cho 70 thực thể
   chưa có bằng bước rút bằng chứng tự động.
3. Lọc lại ảnh tham chiếu: bỏ ảnh qua công cụ AI, ghi URL nguồn, ưu tiên ảnh một chủ thể; thêm Openverse (ảnh CC) làm
   nguồn ảnh có giấy phép.
4. Thiết kế lại review loop (câu hỏi đóng có đối chứng, judge BLIP-2 thay VLM 3B) sau khi ảnh nền ổn; chuẩn bị user
   study trên bộ prompt cuối.
