# CTIG — Bàn giao để đánh giá độc lập

> Bản nháp 1, soạn 2026-09-18. Người soạn là trợ lý AI đã làm việc trên dự án này. Xem mục 10 về xung đột lợi ích.

---

## 1. Đọc cái này trước

**Đề tài:** sinh ảnh đúng văn hoá Việt Nam. Ba nhánh so sánh: (A) prompt tiếng Anh gốc → SDXL; (B) prompt đã
qua Culture-TRIP → SDXL; (C) ảnh của B rồi qua một vòng phê-bình-và-sinh-lại 3 lượt. **B→C là đóng góp
phương pháp duy nhất.** Người làm: một sinh viên HCMUS, làm một mình, nhắm một hội nghị nhỏ.

**Đang mắc ở đâu:** bộ sinh ảnh làm được việc, nhưng **thước đo thì mù**, nên không chứng minh được gì.
Cụ thể, ba con số:

- Thước đo dùng để chọn ảnh có **AUC 0,50** so với nhãn người — đúng bằng đoán mò.
- Hỏi bộ chấm "ảnh nào là ảnh chụp thật": **0,58** (mốc: FAGER 0,97, đồng xu 0,50), và **thiên lệch vị trí 0,42**.
- Bốc thăm 4 seed đang **hoà** với vòng sửa ở cùng ngân sách (thắng 2 thua 1 trên 3 prompt).

**Cần chuyên gia trả lời:** nên chữa thước đo hay bỏ hẳn MLLM làm thước; và đề tài "agentic loop" có còn đủ
sức làm đóng góp chính không. Chi tiết ở mục 8.

**Một lưu ý:** chủ dự án nhận xét trợ lý AI "đang loạn lên và làm tùm lum". Mục 5 và mục 10 trình bày bằng
chứng để chuyên gia tự đánh giá nhận xét đó.

---

## 2. Bối cảnh

Đề tài do thầy hướng dẫn giao. Kho mã công khai: `github.com/OxyzGiaHuy/CTIG`.

Hai đóng góp dự kiến: (1) vòng sửa agentic, (2) bộ 100 prompt văn hoá Việt Nam. Việt Nam hiện vắng mặt trong
các benchmark văn hoá lớn (CultDiff, CuRe, CCUB, CULTIVate), nên đóng góp thứ hai có chỗ trống thật.

Nền tảng lấy từ **Culture-TRIP** (NAACL 2025): nó tinh chỉnh prompt trong không gian văn bản và **không bao
giờ nhìn ảnh**. Khe hở đó là lý do dựng nhánh C. Khung vòng lặp mô phỏng **T2I-Copilot** (arXiv 2507.20536,
ICCV 2025).

Ràng buộc: máy thuê vast.ai theo giờ; 2.343 ảnh tham chiếu do nhóm tải từ web, **giấy phép không rõ, không
phát hành được**; nhãn người hiện chỉ có 12 ảnh do chính chủ dự án gán.

---

## 3. Hệ thống hiện tại

```
prompt VI ─┬─ [A] câu EN gốc ─────────────────────────────→ SDXL ──→ ảnh
           ├─ [B] Culture-TRIP (qwen2.5:14b) ──────────────→ SDXL ──→ ảnh
           │      97–324 từ · 46/47 prompt vượt 77 token CLIP
           └─ [C] ảnh của B là ảnh mốc
                    ↓
              ┌───────────────────────────────────────┐
              │ interpret  → bản phân tích + vật dễ nhầm│
              │ evaluate   → 3 trục, ngưỡng 8,0         │  ×3 vòng
              │ suggestions→ câu mô tả đúng + negative  │
              │ sinh lại (IP-Adapter, 2 ảnh thật)       │
              └───────────────────────────────────────┘
                    ↓ giữ ảnh điểm cao nhất mọi vòng
```

**Công thức chấm** (`ctig/agents/copilot.py`), ba trục, trọng số `prompt 1 : thẩm mỹ 1 : văn hoá 2`:

- `prompt` = trung bình 4 tiểu mục 0–10 (chủ thể có mặt, quan hệ không gian, bám phong cách, nền)
- `aesthetic` = trung bình 4 tiểu mục (bố cục, hoà sắc, ánh sáng, độ nét)
- `culture` = `cultural_fidelity` (0–10, bộ chấm cho thẳng theo thang có neo) − 3 × số chi tiết thuộc văn
  hoá khác, chặn trần bởi `10 × identity_p` nếu `identity_p < 0,5`

Trục thứ ba là phần thêm vào; 10 tiểu mục của T2I-Copilot đều trung tính về văn hoá nên sẽ chấm đạt cho một
chiếc qipao khi prompt nói áo dài.

**Model:** sinh ảnh SDXL 1.0 (bảng phụ: RealVis XL 4.0, FLUX.1-dev). Bộ chấm **Mistral-Small-3.1-24B-Instruct-2503**
chạy local qua transformers — đúng nhánh model mở của T2I-Copilot. Trước đó dùng Qwen2.5-VL-7B nhưng nó trả
danh sách khác biệt **rỗng hoàn toàn** ở câu hỏi nhiều ảnh. Phụ trợ: IP-Adapter, CLIP, OWL-ViT, SAM.

**Chuẩn đối chiếu văn hoá là ảnh chụp thật của nhóm**, không phải bảng kiểm viết tay. Đây là thay đổi lớn
nhất so với phiên bản trước.

---

## 4. Số liệu đã đo — kèm cỡ mẫu

**Cỡ mẫu nhỏ đến mức phải nói trước:** toàn bộ kết luận dưới đây dựa trên **3 prompt** (S001 áo dài, S002
gánh hàng rong, S003 phở) và **12 ảnh**. Phần nhãn người chỉ cho **4 cặp so sánh, tất cả từ một prompt duy
nhất S001** — vì S002 không có ảnh nào người chấm "đúng", S003 không có ảnh nào "sai".

### 4.1 Nhãn người (12 ảnh, mù, xáo thứ tự, `scripts/label_score.py`)

| | kết quả |
|---|---|
| AUC trục văn hoá | **0,75** (thắng 2 · **hoà 2** · thua 0) |
| AUC điểm tổng (trục dùng để CHỌN ảnh) | **0,50** |
| hệ thống giữ ảnh người bảo SAI | **2 / 3 prompt** |
| thuộc tính đạt theo người | 45/51 = **88%** |

Nguyên nhân giữ ảnh sai là **hoà điểm**: ảnh sai và ảnh đúng cùng 5,0 điểm văn hoá và cùng 6,4 điểm tổng, và
luật hiện tại hoà thì giữ ảnh sớm hơn, tức ảnh chưa qua sửa.

Con số 88% đáng chú ý: ảnh S001 vòng 0 (áo có mảng hoa văn đỏ kiểu Trung Quốc) đạt **5/5 thuộc tính** trong
khi người chấm tổng thể là **sai**. Bảng kiểm thuộc tính đã bão hoà.

### 4.2 Best-of-N so vòng sửa, cùng ngân sách 4 ảnh (`scripts/bestofn_control.py`)

| prompt | bốc thăm 4 seed | vòng sửa | |
|---|---|---|---|
| S001 | **6,75** | 6,4 | bốc thăm thắng |
| S002 | **4,75** | 4,6 | bốc thăm thắng |
| S003 | 4,88 | **6,1** | vòng sửa thắng |

Ở S002 và S003, **cả bốn ảnh bốc thăm đều đúng 2,0 điểm văn hoá**, không lệch một phần mười.

### 4.3 Phép thử A/B kiểu FAGER (12 cặp, `scripts/judge_ab_test.py`)

Ghép một ảnh thật với một ảnh sinh, hỏi cái nào là ảnh chụp thật. Hỏi **cả hai thứ tự** rồi lấy trung bình.

Độ chính xác **0,58** · thiên lệch vị trí **0,42**. Mốc: FAGER 0,97 · FineGRAIN 0,83 · VQAScore 0,47.

Thiên lệch mới là điều đáng kể: S002 bốn cặp cho thứ tự 1 là 0,94/0,97/0,96/0,98 và thứ tự 2 là
0,16/0,06/0,02/0,71. **Mistral chủ yếu trả lời "ảnh đứng trước"**. Nếu chỉ hỏi một thứ tự sẽ thu được ~0,90
và kết luận sai rằng bộ chấm rất tốt.

Ngoại lệ: với ba ảnh áo dài đúng kiểu, nó kiên định ở **cả hai thứ tự** rằng ảnh máy vẽ mới là ảnh thật
(0,01 / 0,08 / 0,03) — tức khi phán theo nội dung thì nó phán ngược. Giả thuyết (**chưa kiểm chứng**):
"thật" với nó là ảnh bóng bẩy, mà ảnh SDXL sạch hơn ảnh web.

### 4.4 Preflight 4 phép (`scripts/preflight.py`, chạy sau khi đã sửa mã)

| phép | kết quả | đạt? |
|---|---|---|
| P1 chấm cùng một ảnh 3 lần | lệch **0,00** | đạt — bộ chấm tất định, nên lấy trung bình nhiều lần không cứu được gì |
| P2 hai ảnh THẬT cùng nền văn hoá | trung bình sát 0,5 nhưng **lệch giữa hai thứ tự tới 0,46–0,95** | không |
| P3 `identity_p` sau khi **xáo vị trí** đáp án | vẫn **0,99–1,00** | bão hoà thật, không phải thiên lệch vị trí |
| P4 chấm lại 12 ảnh bằng mã đã sửa | AUC văn hoá vẫn 0,75 với **2 hoà**, AUC tổng vẫn **0,50** | không |

**Điểm văn hoá chỉ nhận ba giá trị {0, 2, 5}** trên toàn bộ 12 ảnh.

Giới hạn của P4: nó chấm lại **ảnh cũ**, nên chỉ kiểm được các bản sửa phía bộ chấm; các bản sửa phía sinh
ảnh (khoá seed, cộng dồn câu mô tả) chưa được kiểm.

### 4.5 Quy luật rút ra từ 4.3 và 4.4

| dạng câu hỏi | kết quả |
|---|---|
| **một ảnh** + phương án bằng chữ | ổn định, xáo vị trí vẫn giữ nguyên |
| **hai ảnh** so nhau | bị vị trí chi phối, lệch 0,42–0,95 |

Điều này **mâu thuẫn** với arXiv 2402.04788 (MLLM làm giám khảo thì so cặp tốt hơn chấm tuyệt đối). Đây là
một trong những câu muốn hỏi chuyên gia.

---

## 5. Lịch sử đổi hướng trong một ngày

43 commit trong ngày 2026-09-17. Trình tự các lần đổi hướng, không bào chữa:

1. Sáng: chuyển máy thuê, dọn 100,5 GB cache trùng do **chính trợ lý tải thừa** (gọi `snapshot_download`
   không lọc → 170 GB thay vì 67 GB).
2. Culture-TRIP chạy xong 100/100 prompt.
3. Dựng cấu hình ba nhánh, chạy lô pilot A và B.
4. **Lỗi trợ lý gây ra:** cờ `--no-grounding` do trợ lý vừa thêm làm mất câu prompt Culture-TRIP, khiến
   nhánh B chạy bằng **câu gốc**. Ảnh nhánh B trùng **byte** với nhánh A ở cả 10 prompt. Log vẫn in
   "prompt ngoài: 97 từ" nên nhìn log không thấy. Phát hiện nhờ so md5. Phải chạy lại.
5. Đổi bộ chấm Qwen2.5-VL-7B → Mistral-Small-3.1-24B.
6. Nhánh C chạy thật lần đầu: 5,5 → 5,9 sau 3 vòng.
7. Sửa 3 lỗi của vòng lặp, chạy lại: 7,4 → 7,6.
8. Đổi công thức trục văn hoá từ **đếm lời chê** sang **chấm theo mức nghiêm trọng**. Chạy lại: hệ thống giữ
   **ảnh sai** ở S001. Tức bản sửa làm kết quả xấu đi.
9. Chủ dự án gán nhãn 12 ảnh → lộ ra AUC điểm tổng 0,50.
10. Chạy ablation best-of-4 → vòng sửa thua 2/3.
11. Chạy phép thử A/B → bộ chấm 0,58, thiên lệch 0,42.
12. Cho 3 agent soi độc lập → tìm ra **việc cắt vùng chủ thể đã tắt im lặng** từ lúc đổi bộ chấm (Mistral
    không có hàm `ground()`), cộng ba lỗi hiệu lực thí nghiệm ở mục 6.
13. Chạy preflight → 1/4 phép đạt.

**Các lỗi do chính trợ lý gây ra rồi phải tự sửa, trong cùng ngày:** tải thừa 100 GB; `--no-grounding` nuốt
prompt (hỏng cả một lô 10 prompt); tắt im lặng việc cắt ảnh khi đổi bộ chấm; `judge_ab_test` nạp Mistral hai
lần gây tràn VRAM; `label_tool` trùng khoá làm 12 ảnh chỉ hiện 6; hai lần tự giết phiên SSH bằng `grep`
khớp vào chính lệnh của mình. **Ít nhất 6 lỗi.**

**Những thứ làm rồi bỏ:** bảng kiểm KB viết tay (bỏ hẳn), chọn ảnh bằng đấu cặp (đo được là chọn đúng ảnh tệ
nhất), câu trắc nghiệm tự sinh từ `must_not` (AUC 0,29, dưới mức đoán mò), `--park-vlm` và `keep_loaded` cho
nhánh C (viết cho card 48 GB rồi quay lại card 80 GB nên không dùng).

---

## 6. Chẩn đoán hiện tại

Tin rằng gốc rễ **không phải** "vòng sửa kém" mà là **chưa bao giờ đo được nó**, vì ba lỗi hiệu lực chồng lên
nhau. Cả ba do một agent thiết kế độc lập tìm ra và đã xác minh bằng mã:

1. **`điểm_C ≥ điểm_B` là đồng nhất thức toán học.** `run_loop` khởi động từ chính ảnh của B rồi trả argmax
   trên tập có chứa nó. Báo cáo con số đó là báo cáo một phép tính.
2. **Nhánh A và B sinh 1 ảnh, nhánh C giữ max của 4.** Max-of-4 trừ trung bình 4 seed, đo trên chính số liệu
   này: +0,55 / +0,11 / +0,10, trung bình **+0,25** — lớn hơn biên thắng thua từng quan sát (+0,35 và −0,15).
3. **`refs` cấp cho IP-Adapter trùng `refs` cấp cho bộ chấm.** Ca duy nhất vòng sửa thắng (S003) vì thế chưa
   loại trừ được lời giải thích "chấm bằng ảnh nó vừa chép".

Độ chắc chắn: ba lỗi trên là **chắc chắn** (đọc mã là thấy). Còn "bộ chấm là nút thắt chính" thì **khá chắc**
nhưng dựa trên 12 ảnh, nên có thể sai.

Quan sát ngược chiều, đáng chú ý: **bộ sinh và vòng lặp làm được việc.** Ở S002 vòng 1, hệ thống chuyển từ
xe đạp chở sọt sang phụ nữ gánh đôi thúng trên đòn gánh — đúng đặc trưng định danh mà vòng 0 thiếu. Ở S001,
từ áo hoa văn Trung Quốc sang áo dài trắng ở sân trường, và nhãn người xác nhận. **Cả hai lần bộ chấm không
nhìn ra**, thậm chí chấm ảnh đúng thấp hơn ảnh sai.

---

## 7. Kế hoạch đang định làm

Bốn giai đoạn, chi tiết ở `docs/KE_HOACH_LOOP.md`:

- **GĐ0** sửa nốt 4 lỗi mã (tách ảnh điều kiện khỏi ảnh chấm, ghi điểm ảnh vòng 0, ghi nhận ca hoà, chạy
  nhiều prompt một tiến trình).
- **GĐ1** đo bộ chấm bằng hai phép không cần nhãn người, trong đó có một phép chưa ai làm: ghép **ảnh áo dài
  thật với ảnh qipao thật** — cả hai đều là ảnh chụp nên mọi manh mối "ảnh máy vẽ" bị triệt tiêu, cái còn lại
  đúng là tri thức văn hoá.
- **GĐ2** lô chính **năm nhánh cùng ngân sách 4 ảnh**, thêm nhánh giả B4 (B với 4 seed, làm sàn nhiễu) và
  nhánh Bp4 (dùng prompt cộng dồn của vòng cuối nhưng **không cho nhìn ảnh**, để tách "phản hồi thị giác"
  khỏi "prompt dài hơn"). 24 prompt, chia dev 9 / test 15, ghi hash danh sách vào git trước khi chạy.
- **GĐ3** một đợt nhãn người duy nhất, ~108 cặp, 45–60 phút.
- **GĐ4** chọn thước trên dev, mở test đúng một lần.

**Chỗ còn nghi ngờ trong chính kế hoạch này:** nó giả định thước đo chữa được. Preflight vừa cho thấy các bản
sửa phía bộ chấm **không cải thiện** số hoà. Nếu thang {0, 2, 5} không phá được thì GĐ2 sẽ cho một bảng không
đọc được, và lẽ ra nên chuyển sang tương đồng ảnh-ảnh sớm hơn.

---

## 8. Câu hỏi cho chuyên gia

**Q1 — Chữa thước đo hay đổi thước đo?**
Bộ chấm được 0,58 ở phép thật-vs-sinh, điểm chỉ nhận {0, 2, 5}, và câu hỏi hai ảnh bị vị trí chi phối.
*Phương án a:* giữ Mistral nhưng hỏi thang mức nghiêm trọng dưới dạng trắc nghiệm **một ảnh** rồi lấy kỳ vọng
trên xác suất các chữ cái (arXiv 2503.03064 báo +7,1 điểm cho cách này ở chấm tuyệt đối). Hợp với quy luật ở
mục 4.5. *Phương án b:* bỏ MLLM khỏi vai thước đo, dùng tương đồng ảnh-ảnh với 2.343 ảnh thật (DINOv2, theo
CultDiff arXiv 2502.08914). Không cần model biết áo dài là gì. *Đánh đổi:* (b) cũng bắt cả tư thế, ánh sáng,
nền — và hiện chưa có cách kiểm chứng nó ngoài chính 12 nhãn người.

**Q2 — Đề tài "agentic loop" có còn đủ sức làm đóng góp chính?**
Best-of-4 đang hoà với vòng sửa; ablation của chính T2I-Copilot cho thấy bỏ Quality Evaluator chỉ mất 0,008;
Ma et al. (arXiv 2501.09732) đo được best-of-N **thắng** tinh chỉnh lặp khi vòng lặp là sinh lại trong không
gian nhiễu. Ngược lại arXiv 2601.15286 và FAGER arXiv 2605.19111 cho vòng lặp thắng đậm — **nhưng cả hai đều
dùng model chỉnh ảnh theo chỉ dẫn**, không sinh lại. Nên: kiên trì với loop, hay đổi đóng góp chính sang bộ
prompt và bộ đánh giá văn hoá Việt?

**Q3 — Có nên đổi hành động sửa sang model chỉnh ảnh (Qwen-Image-Edit, FLUX.1-Kontext)?**
Đây là cấu hình của hai bài thắng cuộc. Nhưng: chưa có thước đo tin cậy để biết nó giúp hay không; và đổi
model sinh giữa vòng 0 và vòng 1+ phá tính "cùng model nền" mà cả thiết kế ba nhánh dựa vào.

**Q4 — Cỡ mẫu tối thiểu là bao nhiêu?**
Hiện 3 prompt, 12 ảnh, 4 cặp từ một prompt. Kế hoạch định lên 24 prompt và ~108 cặp nhãn. Với một sinh viên
làm một mình, mức đó đã đủ để một hội nghị nhỏ chấp nhận chưa?

**Q5 — Mục tiêu thực tế nên đặt ở đâu?**
CulturalFrames (arXiv 2506.08835) đo trần tương quan **người với người** trong đánh giá văn hoá chỉ 0,38, và
mọi thước tự động hiện có nằm ở 0,30–0,31. Trong bối cảnh đó, một kết quả thế nào thì đáng gọi là thành công?

**Q6 — Mâu thuẫn ở mục 4.5.** Tài liệu nói MLLM so cặp tốt hơn chấm tuyệt đối, số đo của dự án lại nói ngược
ở câu hỏi hai **ảnh**. Có phải vì hầu hết tài liệu so cặp là so **văn bản** chứ không phải hai ảnh trong một
lượt? Nếu đúng thì hướng "so cặp" nên bị loại khỏi kế hoạch.

---

## 9. Tài nguyên và cách chạy lại

**Máy:** vast.ai, A800 80 GB PCIe, đĩa 250 GB (còn 113 GB). SSH đổi sau mỗi lần Start.
**Model đã tải** (`/workspace/.hf_home`, 115 GB): SDXL 1.0, RealVis XL 4.0, FLUX.1-dev, Mistral-Small-3.1-24B,
Qwen2.5-VL-7B, IP-Adapter, CLIP, SAM, OWL-ViT.
**Dữ liệu:** `data/prompts_{simple,complex}.json` (50+50 prompt), `data/culture_trip/` (100 file, đã commit),
`docs/labels_huy.json` (12 ảnh, 56 dòng nhãn), `runs_backup/` (167 MB, 134 ảnh + 126 JSON, đã kéo từ máy thuê
về nên không mất khi huỷ instance).
**Ảnh tham chiếu:** 2.343 tấm ở `/workspace/refs_new`, chia `selected/` (3 ảnh mỗi prompt) và `candidates/`
(~20 ảnh). **Chỉ có trên máy thuê**, tải lại được từ Google Drive. Giấy phép không rõ → không phát hành.

**Chạy một prompt đầu-cuối:**

```bash
export HF_HOME=/workspace/.hf_home
export HF_TOKEN=$(grep "^export HF_TOKEN" ~/.bashrc | cut -d= -f2)

# nhánh A và B
python scripts/run_walkthrough.py --config configs/vast_arms.yaml --ids S001 \
    --prompt-source original --no-agents --no-grounding --run-name armA
python scripts/run_walkthrough.py --config configs/vast_arms.yaml --ids S001 \
    --prompt-source culture_trip --no-agents --no-grounding --run-name armB

# nhánh C
python scripts/run_loop_v2.py --config configs/vast_arms.yaml --id S001 \
    --model sdxl_base --prompt-source culture_trip --run-name armC \
    --set llm.backend=mistral_vl --set llm.model=mistralai/Mistral-Small-3.1-24B-Instruct-2503

# các phép đo
python scripts/preflight.py      --config configs/vast_arms.yaml --labels docs/labels_huy.json --runs runs/armC
python scripts/bestofn_control.py --config configs/vast_arms.yaml --ids S001 -n 4
python scripts/judge_ab_test.py  --config configs/vast_arms.yaml --ids S001 --runs runs/armC
python scripts/label_score.py    docs/labels_huy.json runs/armC --axis culture
```

Tài liệu nội bộ: `docs/HANDOFF.md` (nhật ký kỹ thuật đầy đủ, ~940 dòng), `docs/LOOP_v2.md` (thiết kế vòng
lặp), `docs/KE_HOACH_LOOP.md` (kế hoạch 4 giai đoạn).

---

## 10. Xung đột lợi ích — xin đọc

Tài liệu này **do trợ lý AI đã làm phần lớn công việc kỹ thuật soạn ra**, trong khi một phần việc của chuyên
gia là đánh giá chính chất lượng công việc đó. Đây là xung đột lợi ích rõ ràng.

Chủ dự án nhận xét: *"càng làm tôi càng thấy bạn đang loạn lên và làm tùm lum"*. Bằng chứng ủng hộ nhận xét
đó, trình bày ở mục 5: 43 commit trong một ngày; ít nhất 6 lỗi do chính trợ lý gây ra rồi tự sửa, trong đó
một lỗi làm hỏng cả một lô 10 prompt mà không báo lỗi gì; hai lần đổi công thức chấm điểm trong cùng buổi,
lần thứ hai làm kết quả **xấu đi**; và ba lỗi hiệu lực thí nghiệm nghiêm trọng chỉ được phát hiện khi cho
agent độc lập soi, chứ không phải do trợ lý tự tìm ra.

Bằng chứng ngược lại, để chuyên gia cân nhắc: các phép đo phủ định (best-of-N thắng vòng sửa, thước đo AUC
0,50, preflight 1/4) đều do trợ lý tự thiết kế và tự chạy, và đều được báo cáo đầy đủ chứ không giấu.

**Xin chuyên gia đánh giá cả hai điều:** hướng kỹ thuật ở mục 6–8, và liệu cách làm việc phản ánh ở mục 5 có
đang gây lãng phí thời gian của chủ dự án hay không — nếu có thì nên đổi cách làm việc thế nào.

---

*Mọi con số trong tài liệu này lấy từ log và dữ liệu thật trong repo. Chỗ nào là giả thuyết đã ghi rõ
"chưa kiểm chứng".*
