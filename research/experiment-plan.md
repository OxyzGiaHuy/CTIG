# Kế hoạch thực nghiệm CTIG

> Bản kế hoạch thực nghiệm đề xuất. Viết 2026-09-17.

---

## 0. Tóm tắt

| | Thí nghiệm | Câu hỏi | Điều kiện so |
|---|---|---|---|
| **A** | Comparison with Baselines | Hệ thống có hơn các cách làm sẵn có không? | tối thiểu bare vs system; baseline khác chốt sau |
| **B** | Ablation Study | Khối nào đóng góp bao nhiêu? | tắt/bật từng khối; danh sách nhánh chốt sau |
| **C** | User Study | Người Việt có thấy hệ thống tốt hơn không? | A/B mù: bare vs system, cùng model, cùng seed |

**Ba nguyên tắc:**

1. **Cùng seed, ghép cặp.** Wilcoxon signed-rank + bootstrap CI 95%.
2. **Không tự chấm mình.** Metric chính không được đọc `must_have`/`must_not`. Đã code hoá trong [ctig/evaluation.py](../ctig/evaluation.py).
3. **Không rò rỉ ảnh reference.** `selected/<pid>` dành riêng cho phiếu khảo sát; pipeline sinh ảnh không được đọc thư mục này.

---

## 1. Metric tự động

| Trục | Metric | Đo gì | Tương quan với người | Chi phí | Trạng thái |
|---|---|---|---|---|---|
| **văn hoá** | **CultureVQA (bản VN)**<br>[arXiv:2511.17282](https://arxiv.org/abs/2511.17282) | Chỉ nhìn ảnh, VLM chọn 1 trong {VN, TQ, NB, HQ, Tây/chung chung, Không rõ}; báo accuracy | chưa công bố ¹ | rẻ | **cần viết thêm** |
| **văn hoá** | **CAIRe**<br>[arXiv:2506.09109](https://arxiv.org/abs/2506.09109) | Điểm **1–5** độ liên quan văn hoá: mSigLIP + BabelNet liên kết thực thể → Wikipedia → VLM chấm | **r 0,56** (ảnh T2I) ² | trung bình | **cần viết thêm** |
| **văn hoá** | **EXAG** (CULTIVate)<br>[arXiv:2511.05681](https://arxiv.org/abs/2511.05681) | Phóng đại / khuôn sáo: so cường độ yếu tố khuôn sáo với ảnh thật | ρ 0,47 ³ | rẻ | **cần viết thêm** |
| **prompt** | **VQAScore-prompt**<br>[arXiv:2404.01291](https://arxiv.org/abs/2404.01291) | *"Does this figure show `<text_en>`?"* → P(Yes) | 0,30–0,31 ⁴ | rẻ | `prompt_vqa()` đã viết, chưa nối |
| **prompt** | CLIPScore<br>[arXiv:2104.08718](https://arxiv.org/abs/2104.08718) | cosine(ảnh, `prompt_en`) | kém nhất ⁴ | rẻ | **đang chạy** — `clip_prompt_sim` |

¹ Bài gốc chỉ nói pilot "nhất quán cao" với nhãn người, không nêu số. §4 phải tự đo.
² 0,66 trên ảnh thật; tác giả lưu ý chạy tốt trên ảnh thật hơn ảnh sinh. Dùng mSigLIP nên không dính entanglement với SDXL.
³ Là số của FAITH = mean(ALIGN, 1−HAL, 1−EXAG); CULTIVate không báo EXAG riêng. **Descriptor tham chiếu phải sinh độc lập** từ CulturalAtlas/Wikipedia, không lấy từ `data/kb/entities.json`.
⁴ Trên CulturalFrames: trần người–người ρ 0,38; tốt nhất (VIEScore, UnifiedReward) 0,30–0,31; CLIPScore kém nhất trong 6 metric.

**CultureVQA cho ba số**, không chỉ một:

- **accuracy** = % ảnh được gán "Việt Nam" — đây là con số chính.
- **substitution rate** = % gán TQ / NB / HQ → vẽ *sai* sang văn hoá khác.
- **neutralisation rate** = % gán Tây-chung chung / Không rõ → *mất* văn hoá. Suy ra được: `100 − accuracy − substitution`.

**Ba điều kiện bắt buộc khi triển khai:**

1. **Giữ lựa chọn "Không rõ".** Bỏ đi là ép VLM phải đoán, accuracy bị thổi lên.
2. **Hoán vị thứ tự đáp án** rồi lấy trung bình — VLM có thiên lệch vị trí đáp án.
3. **Kiểm đối chứng trên ảnh thật.** Nếu accuracy trên ảnh thật < 0,85 thì VLM không đủ năng lực; đổi model trước khi tin bất kỳ con số nào trên ảnh sinh.

**Đã loại: RefSim** (cosine ảnh sinh ↔ ảnh ref). Ảnh ref là ảnh *thực thể*, ảnh sinh là *cảnh* — cosine bị chi phối bởi nền và góc chụp.

---

## 2. Thí nghiệm A — Comparison with Baselines

**Câu hỏi:** CTIG có hơn các cách tiêm tri thức văn hoá sẵn có không?

Mốc dưới bắt buộc là **bare** (prompt dịch thẳng, cùng model nền, cùng seed). Baseline còn lại và quy mô **chốt sau khi hệ thống final**.

- Metric: văn hoá **CultureVQA**, **CAIRe**, **EXAG**; prompt **VQAScore**, **CLIPScore**.
- Thống kê: Wilcoxon signed-rank trên Δ theo prompt; bootstrap CI 95%; riêng từng model.
- **Một bảng cho simple, một bảng cho complex.** Không tách thêm tầng.

| Model nền | Nhánh | CultureVQA ↑ | CAIRe ↑ | EXAG ↓ | VQAScore ↑ | CLIPScore ↑ | Δ metric chính vs bare (CI 95%) |
|---|---|---|---|---|---|---|---|
| FLUX.1-dev | bare | …% | …/5 | 0,… | 0,… | 0,… | — |
| FLUX.1-dev | **system** | …% | …/5 | 0,… | 0,… | 0,… | **+…** […; …] |
| SDXL 1.0 | bare | …% | …/5 | 0,… | 0,… | 0,… | — |
| SDXL 1.0 | **system** | …% | …/5 | 0,… | 0,… | 0,… | **+…** […; …] |
| … | … | … | … | … | … | … | … |

Δ chỉ tính cho metric chính, các cột khác là mô tả. Substitution / neutralisation cân nhắc tách ra **ma trận nhầm lẫn riêng** (bare vs system, 6 × 6), không làm thành cột.

**Hai cách tách không đưa vào bảng chính:**

- **8 category** — simple chỉ 4–9 prompt/category, CI quá rộng. Để phụ lục, không kèm CI.
- **Độ hiếm thực thể** — chưa có dữ liệu: chỉ 28/50 entity của simple và 25/141 của complex có `prior_strength`. Muốn dùng thì gán tay 22 entity của simple, hoặc dùng proxy bare đúng/sai.

---

## 3. Thí nghiệm B — Ablation Study

**Câu hỏi:** khối nào thực sự đóng góp?

Danh sách nhánh và quy mô chạy **chốt sau khi hệ thống final**.

Đo bằng bộ metric §1. Riêng ở ablation được dùng thêm verifier nội bộ (điểm Filter, CLIP attr, ensemble hạng) để giải thích, vì mọi nhánh đều có KB.

Bảng kết quả sẽ điền — một model nền, Δ tính **so với nhánh full**; Δ âm nghĩa là khối bị tắt có đóng góp:

| Nhánh | CultureVQA ↑ | CAIRe ↑ | EXAG ↓ | VQAScore ↑ | CLIPScore ↑ | Δ metric chính vs full (CI 95%) |
|---|---|---|---|---|---|---|
| **full** | …% | …/5 | 0,… | 0,… | 0,… | — |
| − khối 1 | …% | …/5 | 0,… | 0,… | 0,… | **−…** […; …] |
| − khối 2 | …% | …/5 | 0,… | 0,… | 0,… | **−…** […; …] |
| … | … | … | … | … | … | … |
| bare | …% | …/5 | 0,… | 0,… | 0,… | **−…** […; …] |

Hàng `bare` để làm mốc dưới: khoảng cách full ↔ bare là tổng đóng góp, các hàng ở giữa chia khoảng đó ra.

---

## 4. Thí nghiệm C — User Study

### 4.1 Protocol

```
┌──────────────────────────────────────────────────────────┐
│  Prompt:  "Người phụ nữ Tày mặc áo chàm ngồi hát then…"  │
│  Ảnh thật tham khảo:  [ref 1] [ref 2] [ref 3]            │
├────────────────────────────┬─────────────────────────────┤
│         Ảnh A              │          Ảnh B              │
├────────────────────────────┴─────────────────────────────┤
│ Q1. Prompt alignment — ảnh nào thể hiện đúng nội dung    │
│     câu mô tả hơn?              ( A / B / Ngang nhau )   │
│ Q2. Cultural alignment — ảnh nào đúng văn hoá Việt Nam   │
│     hơn, khi đối chiếu ảnh tham khảo?                    │
│                                 ( A / B / Ngang nhau )   │
│ Q3. (tuỳ chọn) Sai ở đâu?       [____________________]   │
└──────────────────────────────────────────────────────────┘
```

| | |
|---|---|
| Cặp so | bare vs system, cùng model nền, cùng seed |
| Ảnh lấy | ảnh tốt nhất mỗi nhánh theo điểm Reviewer |
| Che thông tin | mù hoàn toàn; trái/phải hoán vị ngẫu nhiên theo seed cố định |
| Thang đo | 2AFC (A / B / Ngang nhau), không Likert |
| Ảnh tham khảo | 3 ảnh từ `selected/<pid>` (chọn tay) |

### 4.2 Ảnh reference

Dùng 3 ảnh `selected/<pid>`, **chỉ cho khảo sát — không đưa vào pipeline sinh ảnh**. Hai điều kèm theo:

1. **Cần đổi cấu hình:** hiện `Session.prompt_refs()` đang lấy `selected/` làm tầng ưu tiên nhất cho IP-Adapter (`retrieval.ref_dir`, `ref_dir_candidates: false`). Phải chuyển pipeline sang `candidates/` hoặc truy hồi web trước khi chạy.
2. **Hướng dẫn chặn so pixel:** *"Ảnh tham khảo cho biết thực thể trông như thế nào. Đừng chấm theo độ giống về bố cục, màu nền hay góc chụp."*

Ghi vào hạn chế: đây là đo độ phù hợp với **một biến thể đã chọn**, nên việc chọn ảnh reference là một phần của phương pháp.

### 4.3 Quy mô

| | Đầy đủ | Tối thiểu |
|---|---|---|
| Prompt | 100 (50 S + 50 C) | 50 (25 S + 25 C) |
| Model | 4 | 2 |
| Số cặp | 400 | 100 |
| Người/cặp | 3 | 3 |
| Tổng lượt đánh giá | 1.200 | 300 |
| Số người | 15 (80 cặp/người) | 6 (50 cặp/người) |
| Thời gian/người | ~35 phút | ~20 phút |

Tiêu chí người đánh giá: sinh ra và lớn lên ở Việt Nam. Ghi thêm quê quán (Bắc/Trung/Nam) và tuổi để tách agreement theo vùng.

### 4.4 Bảng kết quả

| Model nền | Tiêu chí | system | bare | ngang nhau | Win rate system (CI 95%) |
|---|---|---|---|---|---|
| FLUX.1-dev | prompt alignment | … | … | … | …% […; …] |
| FLUX.1-dev | **cultural alignment** | … | … | … | **…%** […; …] |
| SDXL 1.0 | prompt alignment | … | … | … | …% […; …] |
| SDXL 1.0 | **cultural alignment** | … | … | … | **…%** […; …] |
| … | … | … | … | … | … |

Win rate = system / (system + bare), bỏ phiếu "ngang nhau"; CI nhị thức. Lặp lại cho simple và complex.

---

## 5. Những điểm cần chốt

| # | Câu hỏi | Đề xuất |
|---|---|---|
| 1 | Metric **chính** (mang kết luận) là cái nào? | **Chưa chốt** — xem dưới |
| 2 | Encoder cho CLIPScore: CLIP hay SigLIP? | Chạy cả hai một lần, báo độ chênh. Quyết sớm, đổi sau là phải chạy lại |
| 3 | VQAScore-prompt đang nằm trong khoá chọn ảnh — gỡ hay giữ? | Xem dưới |

**(1) Metric chính:**

| | CultureVQA | CAIRe |
|---|---|---|
| Bám mệnh đề "áo dài thành kimono" | trực tiếp, cho luôn ma trận nhầm lẫn | gián tiếp |
| Dễ phát biểu | "sai văn hoá giảm 45% → 12%" | khó hơn, điểm 1–5 |
| Tương quan với người | chưa ai công bố | r 0,56 trên ảnh T2I |
| Chi phí | rẻ | cần mSigLIP + BabelNet |

Đường thứ ba: để §4 đo pairwise accuracy cả hai rồi lấy cái thắng — nhưng thành ra chọn metric sau khi nhìn kết quả, phải khai báo trong bài.

**(2)** CLIPScore là metric duy nhất ở §1 dùng CLIP. SDXL dùng chính text encoder CLIP và huấn luyện trên LAION-2B → generative entanglement. [`CLIPProbe`](../ctig/stages/perception.py) đã nhận SigLIP qua `AutoModel`, đổi chỉ là đổi một khoá config.

**(3) VQAScore-prompt đang không độc lập.** Pipeline tính inline tại [`session.py`](../ctig/session.py) và đưa vào `score_tb()` — khoá sắp xếp cho top-k, ứng viên tốt nhất và **ảnh cuối**:

```
score_tb = (điểm Reviewer, TB VQA thuộc tính, VQAScore-prompt, hạng ensemble)
```

Hai cách: **(a)** gỡ VQAScore khỏi `score_tb`; hoặc **(b)** giữ nguyên nhưng áp cùng khoá chọn cho cả bare lẫn system.

---

## 6. Rủi ro

1. **Chấm bằng bảng kiểm của chính mình** — dễ tái phát mỗi khi thêm metric mới.
2. **Rò rỉ ảnh reference** — pipeline hiện vẫn đọc `selected/` cho IP-Adapter, phải đổi trước khi chạy (§4.2).
3. **FLUX không nhận negative prompt** → Δ nhỏ không có nghĩa hệ thống kém trên FLUX. Phải chú thích.
4. **Người chấm trượt sang so pixel** với ảnh mẫu.
5. **Độ phủ bộ prompt chưa đều** (Bắc Bộ nhiều hơn Trung/Nam; Festival và Customs mỏng) — `data/PROMPTS_README.md`.
