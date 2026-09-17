# Agentic Loop v2 — bỏ KB viết tay, theo khung T2I-Copilot

> Đề xuất 2026-09-17. Chưa chạy. Thay thế toàn bộ vòng lặp v1.9.x.

## 1. Vì sao thay

Vòng lặp hiện tại lấy `must_have_en` / `must_not_en` trong `data/kb/entities.json` làm đích sửa. Ba hệ quả đã
gặp trong hai ngày qua:

- **Phụ thuộc KB viết tay.** Chính tệp đó tự khai "KB hạt giống do Claude soạn để chạy demo… cần đối chiếu
  nguồn thật". KB phủ 37/50 thực thể câu đơn và 33/141 câu phức, phần còn lại phải tự sinh.
- **Phải chỉnh ngưỡng liên tục.** Cả ngày 16-09 xoay quanh việc đặt ngưỡng cho từng thuộc tính: 0,60 cố định
  thì ảnh áo liền quần được +1,00; hiệu chỉnh theo ảnh thật thì S012 còn đúng MỘT thuộc tính kiểm được.
- **Tự chấm mình.** Vòng sửa tối ưu đúng cái mà điểm Reviewer đo, nên hiệu số giữa nhánh có loop và nhánh
  không có loop bị thiên vị ngay từ thiết kế.

## 2. T2I-Copilot làm gì (arXiv 2507.20536, ICCV 2025, MIT)

Ba agent chạy nối tiếp, **không có cơ sở tri thức ngoài, không có bảng must_have/must_not**:

| Agent | Việc |
|---|---|
| **Input Interpreter** `A_in` | hiểu prompt → hỏi lại chỗ mơ hồ → tóm tắt thành **Analysis Report** dạng JSON: elements, attributes, spatial relationships, background, composition, lighting, style |
| **Generation Engine** `A_gen` | nhận diện việc là SINH hay SỬA → chuẩn bị đầu vào (tinh chỉnh prompt, Referring Expression Segmentation khi cần sửa một vật) → chạy model (FLUX.1-dev để sinh, PowerPaint để sửa vùng) |
| **Quality Evaluator** `A_eval` | chấm ảnh trên **10 tiểu mục, mỗi mục 0–10**, rồi so trung bình với ngưỡng |

Mười tiểu mục: *thẩm mỹ* (bố cục, hoà sắc, ánh sáng, độ nét, cảm xúc, độc đáo) và *khớp chữ với ảnh*
(chủ thể có mặt, quan hệ không gian, bám phong cách, nền).

Luật dừng, trích nguyên văn: *"If the average score exceeds the predefined THRESHOLD, the generation is
complete… Conversely, if the score falls below the THRESHOLD… `A_eval` redirects the process to `A_gen`,
incorporating improvement suggestions and user feedback for further refinement."* Ngưỡng 8,0; tối đa 3 vòng.
MLLM dùng GPT-4o-mini.

**Khe hở cho ta:** cả 10 tiểu mục đều trung tính về văn hoá. "Chủ thể có mặt" sẽ chấm đạt cho một chiếc
qipao khi prompt nói áo dài, vì vẫn là "một người mặc áo dài tay cổ đứng". T2I-Copilot không có cơ chế nào
phân biệt áo dài với qipao, kimono hay hanbok.

## 3. Flow đề xuất

```
prompt VI ─► Culture-TRIP ─► prompt* (gốc + mô tả văn hoá)
                                │
                    ┌───────────▼────────────┐
                    │ A_in  Phiên dịch đầu vào│  Analysis Report JSON
                    │  + tư liệu văn hoá      │  (không đọc KB)
                    └───────────┬────────────┘
                                ▼
                    ┌────────────────────────┐
                    │ A_gen Bộ sinh           │  SDXL / RealVis / FLUX
                    │  thang leo 4 nấc        │  + inpaint khi vùng rõ
                    └───────────┬────────────┘
                                ▼
                    ┌────────────────────────┐      đạt
                    │ A_eval Bộ chấm          ├──────────► ảnh cuối
                    │  3 trục, có ẢNH THẬT    │
                    └───────────┬────────────┘
                                │ chưa đạt, kèm góp ý cụ thể
                                └──────► quay lại A_gen (tối đa 3 vòng)
```

### 3.1 `A_in` — Phiên dịch đầu vào

Giữ nguyên ý của họ, thêm một trường. Đầu vào là **prompt gốc + prompt Culture-TRIP + đoạn tư liệu Wikipedia
mà Culture-TRIP đã truy hồi**. Đầu ra là Analysis Report JSON:

```json
{"subjects": [...], "attributes": {...}, "spatial": [...], "background": "...",
 "style": "...", "cultural_entity": {"name_vi": "áo dài", "name_en": "ao dai",
 "look_alikes": ["qipao", "kimono", "hanbok"]}}
```

`look_alikes` **do MLLM suy từ tư liệu truy hồi**, không tra KB. Đây là thứ duy nhất mang tính văn hoá ở bước
này, và nó là đầu ra của bằng chứng chứ không phải danh sách người viết sẵn.

### 3.2 `A_gen` — Bộ sinh, giữ thang leo làm chính sách chọn hành động

T2I-Copilot chỉ có hai hành động (sinh lại / sửa vùng) và để MLLM chọn. Ta giữ khung đó nhưng đặt **thang leo
bốn nấc** vào đúng chỗ "Task Identification", vì đây là phần duy nhất chưa ai công bố (Generation Navigator
2605.17969 chỉ có ba hành động PHẲNG; TARA 2607.18724 định tuyến theo LOẠI lỗi chứ không theo MỨC ĐỘ):

| nấc | khi nào | làm gì |
|---|---|---|
| 1 `edit` | góp ý chỉ về MỘT bộ phận nhỏ và định vị được | inpaint vùng đó, giữ nguyên phần còn lại |
| 2 `ref` | sai về hình dáng tổng thể của thực thể | sinh lại kèm **ảnh thật** qua IP-Adapter |
| 3 `rewrite` | prompt chưa nói rõ chỗ sai | MLLM nhìn ảnh, **vá thêm câu** vào cuối prompt, giữ nguyên phần Culture-TRIP |
| 4 `resample` | đã thử ba nấc trên mà không lên | đổi seed, chỉnh guidance |

Leo nấc khi nấc hiện tại không cải thiện, giống cơ chế `memory` + `patience` đang có. Bỏ ba nấc cũ
`ground_refs` / `attr_refs` / `more_refs` vì chúng khác nhau chỉ ở nguồn ảnh, gộp thành nấc 2.

### 3.3 `A_eval` — Bộ chấm, ba trục

Giữ hai trục của họ, **thêm trục văn hoá**. Mỗi tiểu mục 0–10, MLLM chấm **trên vùng đã cắt quanh chủ thể**.

| trục | tiểu mục | nguồn đối chiếu |
|---|---|---|
| **Khớp prompt** (của họ) | chủ thể có mặt · quan hệ không gian · bám phong cách · nền | chính câu prompt |
| **Thẩm mỹ** (của họ) | bố cục · hoà sắc · ánh sáng · độ nét | ảnh |
| **Văn hoá** (ta thêm) | **1. đúng thực thể** · **2. giống ảnh thật ở đâu, khác ở đâu** · **3. có bị lai văn hoá khác không** | **3 ảnh thật của prompt** + `look_alikes` |

Ba tiểu mục văn hoá hỏi như sau, và đây là chỗ thay thế hoàn toàn bảng must_have/must_not:

1. **Đúng thực thể.** Câu trắc nghiệm ép chọn: *"Vật trong ảnh giống nhất với cái nào: áo dài Việt Nam,
   qipao Trung Quốc, kimono Nhật, hanbok Hàn, hay không rõ?"* Chọn sai hoặc không rõ thì trục này 0.
2. **Khác ảnh thật ở đâu.** Đưa ảnh sinh **cạnh 3 ảnh thật**, hỏi: *"Nêu tối đa 3 khác biệt trên chính vật
   thể, bỏ qua ánh sáng, tư thế và nền."* Danh sách khác biệt này **chính là góp ý gửi sang `A_gen`**, thay
   cho `missing_must_have`. Nó sinh từ ảnh thật, không từ danh sách viết tay.
3. **Lai văn hoá.** *"Có chi tiết nào thuộc văn hoá khác không?"* Có thì nêu tên.

Trung bình có trọng số, **trục văn hoá nhân đôi**, so với ngưỡng. Tối đa 3 vòng như họ.

## 4. Bỏ hẳn những gì

| bỏ | ở đâu |
|---|---|
| `must_have_en` / `must_not_en` làm đích sửa | `CulturalSpec` vẫn tồn tại cho các bước khác nhưng loop KHÔNG đọc |
| hiệu chỉnh ngưỡng từng thuộc tính trên ảnh thật | `describe.calibrate` |
| loại thuộc tính không quan sát được | `describe._CAL_MAX_SPREAD` |
| câu hỏi phủ định đối chứng, câu trắc nghiệm theo thuộc tính | `attr_question_neg`, `FORCED_CHOICE` |
| `garment_rules` luật cứng theo trang phục | `describe.garment_rules` |
| bảy nấc thang cũ | còn bốn |

Giữ lại: `subject_crop` (cắt quanh chủ thể), `inpaint` (nấc 1), IP-Adapter (nấc 2), `qwen_vl.choice_prob`
(dùng cho câu trắc nghiệm "đúng thực thể").

## 5. Thiên lệch còn lại, và cách chặn

Đổi sang khung này **không tự nó hết thiên lệch**: `A_eval` vẫn vừa là đích sửa vừa là thước đo. Khác biệt là
đích sửa nay sinh từ **ảnh thật của chính prompt**, không phải từ danh sách người viết. Ba chốt chặn giữ nguyên:

1. Loop nhìn `selected/`; đánh giá dùng tập ảnh **cất riêng** từ `candidates/`, rời nhau theo băm nội dung
   (`ctig/evaluation.py:ref_split`).
2. Kết luận của bài dựa vào **nhãn người**, không dựa vào điểm `A_eval`.
3. Vẫn cần ablation "loop không nhìn ảnh thật" để tách phần đóng góp của ảnh thật khỏi phần đóng góp của loop.

## 6. Việc phải làm

| # | việc | tệp |
|---|---|---|
| 1 | `A_in` sinh Analysis Report + `look_alikes` từ tư liệu truy hồi | `ctig/agents/interpreter.py` (mới) |
| 2 | `A_eval` ba trục, chấm trên vùng cắt, trả góp ý dạng danh sách khác biệt | `ctig/agents/evaluator.py` (mới, thay `describe.py` trong loop) |
| 3 | `A_gen` thang leo 4 nấc theo góp ý | sửa `ctig/agents/reflector.py` |
| 4 | nối vào `Session.candidate_review` sau `anchor=first` | `ctig/session.py` |
| 5 | kiểm offline bằng agent giả | `tests/test_loop_v2.py` (mới) |

Ước lượng: 1 và 2 là phần nặng, mỗi phần chừng nửa ngày; 3 và 4 nhẹ vì tái dùng được khung cũ.
