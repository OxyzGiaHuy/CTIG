# TÀI LIỆU BÀN GIAO — DỰ ÁN CTIG (sinh ảnh đúng văn hoá Việt Nam bằng pipeline agentic)

**Chốt tại commit `deeeb36`, 2026-09-18.** Repo: `github.com/OxyzGiaHuy/CTIG` (public, nhánh `main`). Mọi lệnh kiểm chứng phải chạy sau `git checkout deeeb36`.

---

## 1. Đọc cái này trước

1. Đề tài: sinh ảnh đúng văn hoá Việt Nam. Ba nhánh cùng seed, cùng SDXL base 1.0: **A** = prompt tiếng Anh gốc; **B** = prompt qua Culture-TRIP; **C** = ảnh của B đi qua vòng lặp sửa 3 vòng kiểu T2I-Copilot.
2. Đóng góp phương pháp duy nhất là **B→C**. Đóng góp thứ hai là bộ 100 prompt văn hoá Việt.
3. **B→C hiện không phải một phép đo**: `run_loop` khởi tạo ảnh tốt nhất bằng chính ảnh B rồi trả argmax, nên `điểm_C ≥ điểm_B` là đồng nhất thức toán học.
4. Trên 3 prompt đã chạy, ảnh cuối của C **trùng byte với ảnh B ở 2/3 prompt**.
5. Thước đo hỏng: điểm văn hoá chỉ nhận ba giá trị {0, 2, 5} trên toàn bộ 23 lần chấm; AUC theo điểm tổng (trục thật sự dùng để chọn ảnh) = **0,50**.
6. Trên 3 prompt có nhãn người, hệ thống giữ lại **0 ảnh đúng / 2 ảnh sai / 1 chưa rõ**.
7. Ablation cùng ngân sách 4 ảnh: bốc thăm 4 seed thắng vòng sửa **2/3**.
8. Toàn bộ phần thực nghiệm dựa trên **3 prompt, 12 ảnh, một người gán nhãn** — chính tác giả.
9. Cần chuyên gia trả lời: **chữa thước hay bỏ MLLM làm thước? vòng lặp còn cửa không? cần bao nhiêu mẫu? cắt gì?** (mục 8).
10. **Tài liệu này do trợ lý AI đã viết phần lớn mã soạn ra — xung đột lợi ích, xem mục 10.**

---

## 2. Bối cảnh và mục tiêu

Một sinh viên HCMUS làm một mình, nhắm một hội nghị nhỏ. Toàn bộ mã trong repo do một trợ lý AI viết (mọi commit đều mang `Co-Authored-By`); không xác minh được dòng nào do người dùng tự tay viết.

**Ràng buộc chưa chốt và cần chuyên gia biết để trả lời có điều kiện:** tên hội nghị, hạn nộp, số giờ máy thuê còn chi được, số giờ người mỗi tuần. Ba trong số các câu hỏi ở mục 8 là câu hỏi phân bổ nguồn lực nên không trả lời được nếu thiếu phần này.

---

## 3. Hệ thống hiện tại

```
data/prompts_simple.json (S001–S050) ──text_en──┬─────────────────────────► NHÁNH A: SDXL, 1 ảnh
                                                │
   Culture-TRIP (NAACL 2025, qwen2.5:14b,       │
   chỉ Wikipedia) → data/culture_trip/*.json ───┼─refined_prompt──────────► NHÁNH B: SDXL, 1 ảnh
                                                │
                                                └─► NHÁNH C = ảnh B + vòng lặp 3 vòng
```

`render: bare` (`ctig/stages/generation.py:116-119`) khiến prompt gửi SDXL đúng bằng **một chuỗi** ở cả ba nhánh; không thuộc tính KB, không LoRA. Negative là hằng số chung.

**Nhánh C** (`ctig/agents/copilot.py`, 579 dòng): A_in diễn giải prompt → vòng 0 sinh SDXL trần (**chính là ảnh B**, đã kiểm md5) → A_eval chấm → A_gen sinh lại kèm góp ý + IP-Adapter Plus scale 0,5. Không có thang leo, không có inpaint, hành động sửa duy nhất là sinh lại toàn bộ ảnh.

**Công thức chấm** (`copilot.py:25, 266-291`):

```
overall = (prompt + aesthetic + 2·culture) / 4
culture = cultural_fidelity  (VLM tự cho, thang neo 0/2/5/8/10)
          nếu identity_p < 0,5:  culture = min(culture, 10·identity_p)
          culture -= 3·len(foreign_elements),  kẹp [0,10]
dừng khi: overall ≥ 8,0  VÀ  foreign rỗng  VÀ  differences rỗng
chọn cuối: argmax(overall) trên cả 4 ảnh, so sánh chặt `>` (copilot.py:540, 566, 578)
```

**Model:** SDXL base 1.0 + VAE fp16-fix (cả ba nhánh); IP-Adapter Plus ViT-H (chỉ C, vòng ≥1); bộ chấm **Mistral-Small-3.1-24B-Instruct** — `configs/vast_arms.yaml:21` vẫn ghi `qwen_vl`, phải truyền tay `--set llm.backend=mistral_vl`; OWL-ViT cắt vùng chủ thể. Bộ chấm **không có trong `scripts/vast/dl_models.sh`** (9 kho, không kho nào là Mistral) — một lỗ tái lập.

**Ba lỗi hiệu lực thí nghiệm, đều còn nguyên ở HEAD:**

1. `điểm_C ≥ điểm_B` là đồng nhất thức (argmax khởi từ ảnh B).
2. Ngân sách lệch: `vast_arms.yaml:41` đặt `n_candidates: 1` cho A/B, C giữ max của 4. Chính dòng chú thích `vast_arms.yaml:15` tự khai "mỗi nhánh đúng một ảnh, cùng seed" — mã nhánh C vi phạm chú thích của chính nó.
3. Nhiễm chéo: cùng biến `refs` vừa cấp cho IP-Adapter (`run_loop_v2.py:138`) vừa làm chuẩn chấm (`:155`).

**Bất đối xứng thứ tư, chưa từng nêu:** prompt Culture-TRIP dài trung vị 208 token CLIP, **99/100 prompt vượt 77 token**, nên B/C đi qua đường nối embedding bằng compel còn A thì không. A và B khác nhau cả ở cơ chế điều kiện văn bản, không chỉ nội dung câu.

---

## 4. Số liệu đã đo

**Nhãn người.** 56 dòng nhãn thuộc tính trên **12 ảnh, 3 prompt, một người gán** (`annotator: "huy"`, chính tác giả). Theo *ảnh*: 5 sai / 4 đúng / 3 không chắc. Thời gian từng ảnh: 66,4 · 61,5 · 48,7 · 31,3 · 25,9 · 21,0 · 18,9 · 17,3 · 15,7 · 7,6 · 4,6 · 3,3 giây — ba ảnh cuối gán dưới 8 giây. Ảnh hiển thị là thumbnail 760 px JPEG q80.

**Cảnh báo quan trọng về ý nghĩa nhãn:** câu hỏi tổng thể trong `scripts/label_tool.py:89` là *"Tổng thể: ảnh này có đúng là **{subject}** không?"* với subject = danh sách thực thể ("áo dài"). Đây là câu hỏi **định danh thực thể**, không phải câu hỏi về tính đúng văn hoá. **Dự án chưa có một nhãn người nào hỏi về văn hoá.** Bảng kiểm thuộc tính cũng do đường KB tự sinh, S001 có 5 must_have / 0 must_not và **không mục nào hỏi về hoa văn**.

**"AUC" ở đây** := (thắng + 0,5·hoà)/tổng cặp, cặp chỉ ghép trong cùng prompt. **Không phải ROC-AUC.**

| Con số | Cỡ mẫu thật | Tái lập được? |
|---|---|---|
| AUC trục văn hoá **0,75** (2 thắng, **2 hoà**, 0 thua) | **4 cặp, tất cả từ 1 prompt (S001)** | Có |
| AUC điểm tổng **0,50** (2 thắng, 0 hoà, 2 thua) | cùng 4 cặp | Có |
| Ảnh hệ thống giữ: **0 đúng / 2 sai / 1 chưa rõ** | 3 prompt | Có |
| Thuộc tính 45/51 = 88 % | 51 phán đoán, 11 ảnh | Có, nhưng bão hoà |
| Ablation best-of-4: **1 thắng 2 thua** | 3 prompt | Có, khớp từng chữ số |
| Điểm văn hoá chỉ nhận {0, 2, 5} | **23 lần chấm** | Có — chắc nhất |
| A/B kiểu FAGER acc 0,58 · lệch vị trí 0,42 | 12 cặp | **Không** — không có file kết quả |
| Preflight 4 phép | 12 ảnh | **Không** — không có file kết quả |

S002 không có ảnh nào nhãn "dung", S003 không có ảnh nào nhãn "sai" → **hai prompt đó đóng góp 0 cặp**.

**Bóc tách AUC 0,75.** Hai "thắng" đều là so với iter1 của S001 — ảnh duy nhất bị trừ −6 điểm vì hai mục "western-style". Hai "hoà" là iter0 so iter2/iter3. Đáng chú ý: ảnh iter0 (áo hoa văn Trung Quốc, người chấm SAI, hệ thống **giữ**) được fidelity 5,0. Bộ chấm **có** nhìn thấy hoa văn — mục thứ ba trong `differences` là `"right side panel has floral pattern instead of being plain"` — nhưng bộ lọc `_names_a_culture` (`copilot.py:82-84`, danh sách `_CULTURES` 43 từ) chỉ cho một mục vào ô `foreign` nếu nó **tự đặt tên một nền văn hoá**. "floral pattern" không nêu tên nền nào nên đi qua miễn phí. Đây là lỗi định vị được và sửa được.

**23 lần chấm** (11 pilotC + 12 bestof4; eval ảnh iter0 của S003 không được ghi): `cultural_fidelity` chỉ nhận 2,0 (×16) và 5,0 (×7); ba neo 0/8/10 chưa bao giờ dùng. Trục văn hoá {0,0 ×1; 2,0 ×15; 5,0 ×7}. `identity_p` ∈ [0,9973; 0,9999] — hằng số thực tế. `differences` trả đúng 3 mục ở **21/23** lần. `foreign` khác rỗng **1/23**.

**Hai trục kia gần như là hằng số.** S001 trục prompt = 7,25 cả ba vòng; trục thẩm mỹ dao động ≤ 0,5. Phương sai điểm tổng gần như toàn bộ đến từ trục văn hoá ba mức nhân hệ số 2.

**Ablation best-of-4** (cùng ngân sách 4 ảnh, seed hiệu dụng 1234/2235/3236/4237 do `bestofn_control.py:51` đặt cả `seed` lẫn `iteration` rồi `generation.py:356` cộng `1000·iteration`):

| prompt | 4 seed | best-of-4 | vòng sửa | biên |
|---|---|---|---|---|
| S001 | 6,44 · 6,69 · 4,94 · **6,75** | 6,75 | 6,44 | **+0,31** bốc thăm |
| S002 | 4,62 · 4,56 · 4,62 · **4,75** | 4,75 | 4,62 | **+0,13** bốc thăm |
| S003 | **4,88** · 4,88 · 4,81 · 4,56 | 4,88 | 6,06 | **−1,19** vòng sửa |

Lợi thế "miễn phí" của max-of-4 so với trung bình 4 seed: **+0,545 / +0,113 / +0,098, trung bình +0,25**. Con số này đủ giải thích hai ca bốc thăm thắng nhưng **không** đủ giải thích ca thứ ba. Tuy nhiên ca thắng duy nhất đó **không được nhãn người ủng hộ**: ở S003 hệ thống giữ iter2 (người: *không chắc*) và bỏ iter3 (người: *đúng*, máy chỉ cho 4,4). Ở S002 và S003, cả 4 ảnh bốc thăm đều đúng 2,0 điểm văn hoá — không lệch một phần mười.

**Vòng lặp không tích luỹ và có vòng rỗng.** Trích thẳng `loop_v2.json`:
- S001 iter2 bị chê "sleeves are too short"; iter3 — sinh ra để sửa đúng lời đó — bị chê "sleeves are too long".
- S002 iter1 "attached to a single pole instead of a yoke" → iter2 "attached to a yoke instead of a carrying pole" (tự đảo chiều).
- **S002 iter3: `positive = ""`, `negative = []`** — một vòng sửa rỗng, tức 1/3 ngân sách sửa của prompt đó chỉ là một lần bốc thăm lại.
- **Nhiễm chéo có hậu quả đo được:** S001 iter0 bị chê "left side panel is red instead of white", câu sửa gửi SDXL là *"Áo dài with left side panel white, right side panel **red**"* — trong khi prompt yêu cầu **áo dài trắng**. Bộ chấm đang ép bộ sinh khớp ảnh tham chiếu, không khớp prompt.

**Ba con số PHẢI coi là chưa kiểm chứng:** mọi kết quả preflight (P1–P4) và mọi con số phép thử A/B chỉ tồn tại dưới dạng chữ trong `docs/HANDOFF.md`; không có file kết quả nào trong repo. Mọi mốc lấy từ bài báo khác (FAGER 0,97 · VQAScore 0,47 · T2I-Copilot 0,813→0,805 · CulturalFrames 0,38 · ImageReward 1,58 vs 1,49) **chưa ai trong nhóm kiểm lại tại nguồn**.

**P1 như đang viết là vô nghĩa.** `configs/vast_arms.yaml:21` đặt `cache: true`; khoá đệm là sha1 **nội dung ảnh** + system + user (`ctig/llm/cache.py:43-49`), thư mục đệm dùng chung mọi run. `scripts/preflight.py` chấm lại cùng một ảnh mà không tắt đệm → lần 2-3 là cache hit, "lệch 0,00" được bảo đảm bởi mã. Tương tự, việc hai lô chấm cùng ảnh trùng byte cho **điểm và chuỗi differences giống nguyên văn từng chữ** là dấu hiệu cache hit, **không** phải bằng chứng bộ chấm tất định. Kéo theo: bản vá `subject_crop` ở `539550e` chưa được chứng minh là đã có hiệu lực — nếu nó thật sự cắt khác đi thì khoá đệm đã khác. **Kết luận đúng: 0/4 phép preflight cho kết luận dùng được, và tính tất định của bộ chấm chưa được đo.**

**Nhánh A vs B: không có một điểm chấm nào.** `runs_backup/pilotA3` và `pilotB3` mỗi thư mục 10 prompt × 3 model nền, chỉ có `multigen.json` và `step_*.json`. Nhận xét A-vs-B trong HANDOFF là quan sát bằng mắt. Điểm tích cực: md5 cho thấy A ≠ B ở cả 10 prompt, tức lô đang giữ là lô **đã chạy lại sau bản vá** `--no-grounding`, không phải lô hỏng.

---

## 5. Lịch sử đổi hướng — trung thực

Cửa sổ 2026-09-16 → 09-18: **141 commit** (77 · 60 · 4 theo cả author date lẫn committer date).

- **09-16 13:16** Bỏ KB viết tay → KB tự sinh. Không có bằng chứng số. 14 commit liên tiếp vá bộ sinh KB.
- **09-16 20:45→21:26** Ba lý thuyết khác nhau cho cùng một triệu chứng trong **41 phút**; hai cái đầu sai. Câu trắc nghiệm hai lựa chọn sống **13 phút** (AUC 0,29 — dưới ngẫu nhiên).
- **09-16 23:33** Kết luận: **không được cho một nhánh có bộ chọn mà nhánh kia không có**. Bài học này bị đánh mất 18 giờ sau — đó chính là lỗi hiệu lực số 2.
- **09-17 13:50** Bỏ hẳn KB, dựng lại theo T2I-Copilot. Lý do: *"theo yêu cầu người dùng"*, không phải một phép đo. ~31 commit của ngày 09-16 rời khỏi đường chạy.
- **09-17 14:36→15:25** Chọn ảnh cuối bằng đấu cặp vòng tròn: thêm rồi gỡ trong **49 phút** (chọn đúng ảnh tệ nhất). Mã chết còn trong repo (`copilot.py:486, 509`). Commit gỡ ghi *"Hiểu nhầm yêu cầu"* — **đây là lỗi của trợ lý, không phải yêu cầu sai**.
- **09-17 17:41→22:37** Đổi bộ chấm sang Mistral làm `subject_crop` thành no-op **295 phút**, đúng bước từng đưa AUC 0,67→0,82. Commit sửa (`539550e`) thực ra sửa **ba** lỗi, hai lỗi kia chưa từng được nêu: đường lùi đẩy lời chê **thô** vào negative prompt (cấm đúng thứ cần vẽ: "woven bamboo", "circular"), và bộ lọc phủ định so chuỗi con nên "a kimono sleeve" bị loại vì trong "kimono " có "no ".
- **09-17 19:36** Viết lại trục văn hoá, **346 phút** sau khi chính trợ lý viết công thức đó.
- **09-17 22:34** `c6dc1e1` phát hiện **prompt dương KHÔNG cộng dồn** (negative thì có) — vòng 3 mất sạch phần vòng 1-2 đã sửa. Cộng với việc mỗi vòng một seed: **bốn ảnh của lô pilotC là bốn mẫu độc lập, không phải bốn bước tinh chỉnh.** Mọi kết luận về vòng sửa đo trên một hệ thống chưa từng tích luỹ.
- **09-17 23:55 → 09-18 00:27** Trợ lý bác giả thuyết của chính mình về `identity_p` (nói là thiên lệch vị trí), rồi giả thuyết mới cũng sai (bão hoà thật). Hai lần đoán sai liên tiếp cùng một con số.
- **09-18 00:04** Ba lỗi hiệu lực được tìm ra bởi các **phiên con của cùng một mô hình AI** do chính trợ lý khởi chạy — độc lập về **ngữ cảnh**, không độc lập về mô hình hay lợi ích. **Không có người thứ hai nào đọc mã.** Tài liệu bạn đang đọc cũng sinh ra bằng cơ chế đó.

**Đếm lỗi tự gây.** Phân loại tay: **ít nhất 38** commit sửa một hành vi sai cụ thể do chính trợ lý viết trong cùng cửa sổ (bao gồm `cd5d441` — script tải model gọi `snapshot_download(allow_patterns=None)` cho 8/9 kho, kéo 170 GB thay vì ~67 GB — và `d6c5b9f` — tạo Session mới mỗi prompt làm nạp thêm một bản Mistral 48 GB, tràn VRAM). Chỉ **5** commit sửa nguyên nhân bên thứ ba. Phép đếm bằng `git blame` cho **63–73/135** tuỳ cách cài; script không được commit nên **con số này hiện không tái lập được** như đã hứa.

**Thời gian sống của lỗi:** 0 · 1 · 2 · 4 · 5 · 12 · 32 · 49 · 295 · 346 phút.

**Phải khai báo:** một bản bàn giao trước đó đã nằm trong git (`docs/BAN_GIAO_CHUYEN_GIA.md`, commit `08660ad`, 00:48 cùng đêm), gửi cùng mục đích, và nói *"ít nhất 6 lỗi"* cùng *"43 commit một ngày"*. **Cả hai con số đó sai thấp**: git cho 60 commit ngày 09-17 và phép đếm ở trên cho ít nhất 38 lỗi. Hai con số đó do chính trợ lý viết.

**Chi phí đo được của lỗi tự gây:** một lô pilot 10 prompt × 3 model phải chạy lại; ~100 GB đĩa thuê tải thừa; lô pilotC 3 prompt chạy khi `subject_crop` là no-op **và** prompt dương chưa cộng dồn.

**Không có dấu hiệu che giấu** — nhưng nhận định này chỉ dựa trên nội dung commit message; không kiểm được có thất bại nào xảy ra mà không được ghi lại.

---

## 6. Chẩn đoán hiện tại và độ chắc chắn

| Chẩn đoán | Độ chắc |
|---|---|
| Thang điểm văn hoá quá thô để phân biệt (3 mức, hoà là mặc định) | **Cao** — 23 lần chấm, không phụ thuộc nhãn người |
| Hình phạt văn hoá chỉ kích hoạt khi VLM tự đặt tên nền văn hoá; hoa văn lai đi qua miễn phí | **Cao** — định vị được ở `copilot.py:82-84` |
| Vòng lặp chưa bao giờ tích luỹ (prompt dương không cộng dồn, seed đổi, có vòng rỗng) | **Cao** — đọc được trong mã và dữ liệu thô |
| Thiết kế thí nghiệm hiện tại không thể sinh ra một kết luận hợp lệ về B→C | **Cao** — ba lỗi hiệu lực còn nguyên |
| Bộ chấm không phân biệt được ảnh thật/ảnh sinh | **Thấp** — 12 cặp, câu hỏi có lỗi (`judge_ab_test.py:88` chèn cả câu prompt vào chỗ tên thực thể), không có file kết quả |
| Vòng sửa thua best-of-N | **Rất thấp** — n=3, thước mù, ca thắng duy nhất không được nhãn người ủng hộ |
| Bộ chấm tất định | **Chưa đo được** — cache bật |

**Nói thẳng: ở trạng thái hiện tại, dự án chưa có một kết quả thực nghiệm nào về hiệu quả của phương pháp.** Thứ nó có là một chẩn đoán khá chắc về thước đo.

**Một rủi ro hiệu lực chưa từng gộp lại thành một hình:** prompt nhánh B do qwen2.5:14b viết; bảng thuộc tính để gán nhãn do LLM + web sinh; danh sách "vật dễ nhầm" do Mistral sinh; điểm do chính Mistral chấm; tri thức văn hoá chỉ từ Wikipedia. Người duy nhất ngoài vòng tròn là tác giả, gán 12 ảnh bằng một câu hỏi về **định danh**, không về văn hoá.

---

## 7. Kế hoạch đang định làm (và chỗ nghi ngờ)

`docs/KE_HOACH_LOOP.md`: 9 bản sửa (12–16 giờ công) → hai phép thử verifier (6–8 giờ công, 2–4 giờ máy) → lô chính 5 nhánh × 24 prompt (10–16 giờ máy) → một đợt nhãn người (~108 cặp, 45–60 phút) → chọn thước và viết (~30 giờ). Tổng ~40 giờ công + ~20 giờ máy. Kế hoạch chốt **không nhánh nào dùng IP-Adapter** và **cấm DINOv2 vào vòng chọn**.

Chỗ nghi ngờ: (a) mọi ước lượng giờ là tự khai, chưa đối chứng; (b) kế hoạch chưa có **giao thức chấm cho nhánh A và B** — mà khi không có ảnh thật thì mã **bỏ hẳn trục văn hoá** và điểm tổng đổi từ 4 trọng số xuống 2, tức điểm hai nhánh **không so được với nhau**; (c) `docs/KE_HOACH_LOOP.md` còn chép sai một con số ("85 file culture_trip"; thực tế 100) và một biên ablation không tồn tại trong dữ liệu ("−0,15").

---

## 8. Câu hỏi cho chuyên gia

**C1 — Chữa thước MLLM, hay bỏ MLLM khỏi vòng chọn?** Điểm chỉ có 3 bậc, AUC theo trục chọn = 0,50, và phép thử thật-vs-sinh có lỗi câu hỏi nên 0,58 chưa phải mốc cuối. Phương án A: chữa (đổi sang nhiều câu có/không trên MỘT ảnh? thang neo bằng chữ? so cặp?). Phương án B: DINOv2/CLIP làm hàm chọn, MLLM chỉ viết lời phê — nhưng khi đó mất tín hiệu duy nhất không do bộ chấm sinh ra. Chống lại A: CulturalFrames báo mọi thước tự động hiện chỉ đạt 0,30–0,31 (**chưa kiểm tại nguồn**).

**C2 — Vòng lặp có còn đủ sức làm đóng góp chính?** Dữ liệu: 1 thắng 2 thua trên n=3 với thước mù; C ≡ B ở 2/3 prompt; ablation của chính T2I-Copilot cho thấy bỏ Quality Evaluator chỉ mất 0,008. Giữ, đổi trọng tâm sang "đo verifier", hay hạ xuống mục phụ? "Đo verifier" có đủ nặng cho hội nghị nhỏ?

**C3 — Cỡ mẫu tối thiểu để một phản biện chấp nhận một kết luận, kể cả kết luận âm?** Nguyên liệu không thiếu (100 prompt đã sẵn); **nhãn người là nút thắt**. Có chấp nhận được không nếu tác giả tự gán phần lớn và nhờ 1–2 bạn gán ~25 cặp trùng để báo độ đồng thuận?

**C4 — Đổi hành động sửa sang model chỉnh ảnh theo chỉ dẫn (Qwen-Image-Edit / FLUX.1-Kontext)?** Hai bài duy nhất mà vòng lặp thắng best-of-N đều dùng hành động sửa kiểu đó, và FAGER thắng lớn chỉ với **một vòng**. Nhưng cả hai bài đó đã có thước dùng được; dự án này chưa. Thứ tự đúng là gì? Và nếu một vòng là đủ thì kiến trúc "3 vòng" còn lý do gì?

**C5 — Trần tương quan người-người cho đánh giá văn hoá chỉ 0,38.** Bài nên hứa gì: cải thiện trên thước tự động, tỉ lệ thắng theo cặp do người chấm, hay một giao thức đo + một kết quả âm trung thực? Bao nhiêu phần trăm trên bao nhiêu cặp thì đủ để không bị gạt?

**C6 — Sau khi sửa ba lỗi hiệu lực, dữ liệu cũ vô hiệu hoàn toàn hay còn dùng được phần nào?** Nếu vô hiệu thì 12 nhãn người cũng hết hạn (nhãn đo trên ảnh do mã cũ sinh). Lỗi nào bắt buộc phải sửa trước khi tiêu thêm một giờ máy?

**C7 — Bộ 100 prompt song ngữ, KHÔNG kèm ảnh tham chiếu và KHÔNG kèm nhãn người, có tính là một đóng góp?** Prompt do nhóm tự viết, chưa có tài liệu nào ghi một vòng kiểm chéo của người bản ngữ ngoài nhóm. Cần thêm tối thiểu gì? Xử lý bản quyền 2.343 ảnh tham chiếu thế nào (chỉ phát hành URL + hash? chỉ đặc trưng DINOv2?).

**C8 — So cặp hay chấm tuyệt đối?** Tài liệu nói MLLM so cặp tốt hơn; số đo của dự án lại cho thấy câu hỏi HAI ảnh bị vị trí chi phối (0,42) còn câu hỏi MỘT ảnh thì ổn định. Nhưng phép đấu cặp thất bại chạy trên **Qwen-VL 7B**, chưa chạy lại bằng Mistral 24B. Có đáng chi thời gian chạy lại không?

**C9 — Nếu chỉ còn một phần ba ngân sách, giữ bước nào?** Cụ thể: phép thử nào có khả năng cao nhất **lật ngược cả hướng nghiên cứu** và vì thế phải chạy trước tiên?

**C10 — Mức tối thiểu về tách vai trò để phần đánh giá được chấp nhận?** Hiện người viết mã, người chọn ảnh, người gán nhãn và người viết bài là cùng một người. Có nên nói thẳng trong bài rằng phần lớn mã do một trợ lý AI viết không?

---

## 9. Tài nguyên và cách chạy lại

**Hai môi trường, không cái nào tự đủ.** Máy cá nhân: WSL2 **không GPU**, chỉ chạy được chế độ giả lập. Máy thuê vast.ai: 1× A800 80 GB, đĩa 250 GB, `/workspace/ctig17`. **Chưa xác minh máy thuê còn sống hay đã bị Destroy** — mọi thông tin về `/workspace` đọc lại từ `docs/HANDOFF.md`, không ssh kiểm.

**Trong git:** 100 prompt (`data/prompts_*.json`), 100 file Culture-TRIP, KB (86 KB), 48 script `scripts/vast/`, và từ `deeeb36` là **12 nhãn người** (`data/labels/labels_huy.json` — trước đó untracked).

**KHÔNG trong git:** `runs_backup/` — **167 MB, 134 ảnh PNG**, toàn bộ bằng chứng ảnh của mọi con số. `.gitignore:11` viết `runs_backup/   # chú thích` mà git không coi `#` giữa dòng là chú thích, nên mẫu **không chặn gì**: `git check-ignore -v runs_backup/` trả rỗng, `git status` hiện `?? runs_backup/`. 167 MB đang không được bảo vệ trong một repo công khai.

**KHÔNG có ở đâu ngoài máy thuê:** 2.343 ảnh tham chiếu (897 MB), **giấy phép không rõ, không có URL nguồn từng ảnh** (`download_log.jsonl` chỉ ghi số lượng → không truy ngược được), và một phần đã qua công cụ AI (prompt p001 có 4/10 ảnh tên `dreamina-...`, còn watermark trang bán áo dài). **Mọi kết quả nhánh C hiện có đều phụ thuộc vào dữ liệu không phát hành được.**

**Chạy lại không cần GPU (đã chạy thật, 5 giây):**
```bash
git checkout deeeb36 && python3 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python scripts/run_walkthrough.py --config configs/offline.yaml --ids S001 \
  --prompt-source culture_trip --set prompts_path=data/prompts_simple.json --run-name smoke
```
Mọi model là stub — chỉ kiểm được luồng dữ liệu.

**Kiểm chứng tài liệu này trong 10 phút** (cần `runs_backup/`, phải xin kèm):
```bash
python3 scripts/label_score.py docs/labels_huy.json runs_backup/pilotC --axis culture
#  → 4 cặp · 2 thắng 2 hoà 0 thua · AUC 0.75 · giữ: 0 đúng 2 sai 1 chưa rõ
python3 scripts/label_score.py docs/labels_huy.json runs_backup/pilotC --axis overall   # → AUC 0.50
md5sum runs_backup/pilot{A3,B3}/S001/sdxl_base_rbare/*.png runs_backup/pilotC/S001/iter0/sdxl_base/*.png
#  → B và C-iter0 trùng f190925285b7fca60d9eef28a13a8809 ; A khác (b0e0a854...)
git check-ignore -v runs_backup/ ; echo "exit=$?"   # → rỗng, exit=1
git log --format='%ad' --date=short | sort | uniq -c | tail -3   # → 77 / 60 / 4
```

**Phải gửi kèm tài liệu này** (vài trăm KB, hiện chưa đính): 3× `loop_v2.json`, 3× `bestofn.json`, 3× `step_spec.json`, `labels_huy.json`, và **ảnh** — ba lưới 4 vòng của S001/S002/S003 có chú thích nhãn người + điểm máy, ba ảnh `selected/` mỗi prompt, ảnh nhánh A và B cùng ba prompt, và riêng ảnh S001/iter0 phóng to. **Đây là tài liệu về một đề tài sinh ảnh mà chưa có một tấm ảnh nào** — nếu thiếu, C1/C2/C5 không trả lời có trách nhiệm được.

---

## 10. Xung đột lợi ích

Tài liệu này do **trợ lý AI đã viết phần lớn mã trong repo** soạn ra, để gửi cho một chuyên gia đánh giá độc lập — trong đó có việc đánh giá chất lượng công việc của chính trợ lý.

Bối cảnh: người dùng vừa nói **"càng làm tôi càng thấy bạn đang loạn lên và làm tùm lum"** rồi yêu cầu soạn tài liệu này. Nhận xét đó **có cơ sở trong dữ liệu**, cụ thể ở mục 5: ít nhất 38 commit là sửa lỗi tự gây trong cùng ba ngày; ~31 commit của trọn một ngày làm việc rời khỏi đường chạy; hai tính năng bị xây rồi gỡ trong 13 và 49 phút; ba lý thuyết cho cùng một triệu chứng trong 41 phút; một bài học rút ra lúc 23:33 bị đánh mất 18 giờ sau và trở thành đúng một trong ba lỗi hiệu lực; và ba lỗi hiệu lực nặng nhất không do trợ lý tự tìm ra sau 43 giờ làm việc trên chính mã đó. Thêm nữa, **bản bàn giao trước đó — cũng do trợ lý soạn, cũng cho chuyên gia — đã báo "ít nhất 6 lỗi" và "43 commit", cả hai đều sai thấp.**

Cách xử lý đã áp dụng: mọi khẳng định đều kèm `file:dòng` hoặc mã băm commit; các lỗi nặng nhất được nêu **trước** mọi kết quả tích cực; mọi con số bị nghi ngờ đã được chạy lại trực tiếp trên mã và dữ liệu và đã sửa khi lệch (đáng kể nhất: 24 → 23 lần chấm; "2/3 giữ ảnh sai" → "0 đúng / 2 sai / 1 chưa rõ"; "bộ chấm tất định" → chưa đo được vì cache bật; "biên thắng thua 0,11–0,31" → biên lớn nhất thật sự là −1,19 và nó chống lại lập luận).

**Mời chuyên gia đánh giá cả điều này:** liệu một quy trình có nhịp sửa lỗi như mục 5 có thể cho ra một kết quả nghiên cứu đáng tin không, và nếu có thì cần cổng kiểm soát nào (đăng ký trước giao thức? hash danh sách test vào git trước khi chạy? người gán nhãn thứ hai bắt buộc?).

---

*Điều tôi đã KHÔNG kiểm: không chạy lại bất kỳ bước GPU nào; không ssh vào máy thuê; không đọc mã Culture-TRIP gốc; không đọc lại bài tham khảo tại nguồn; không xem một tấm nào trong 134 ảnh; không có người thứ hai xem nhãn.*

*Một điểm trong các lượt soi nội bộ mà tôi bác bỏ: giả thuyết rằng chênh lệch "75 vs 77 commit ngày 09-16" là do author date khác committer date — tôi đếm cả hai và cả hai đều cho 77; con số 75 đơn giản là sai, không phải một khác biệt về loại ngày.*