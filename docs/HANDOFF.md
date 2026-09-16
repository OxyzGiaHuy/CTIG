# HANDOFF — CTIG, trạng thái 2026-09-16 (cập nhật liên tục sau mỗi lượt làm việc)

Tài liệu bàn giao cho người tiếp tục (hoặc cho chính nhóm sau khi mở lại máy). Đọc cùng `docs/FLOW_v1.7.md` (flow và model
từng bước), `research/research-log.md` (nhật ký theo ngày), `research/research-state.yaml` (giả thuyết H1–H21).

## 1. Phiên bản code và ý nghĩa

| tag | commit | nội dung | trạng thái chạy |
|---|---|---|---|
| v1.7 | `5b770e6` | Grounding gom bước; hàng `M#bare` cùng seed; Agentic Review Loop (Reviewer / Reflector / Refiner, 3 vòng, giữ hết ảnh, chọn trên pool) | đã chạy 4 prompt gốc + 4 complex trên A100 |
| v1.7.1 | `862df1c`… | Reviewer: VQA có/không từng thuộc tính, VQAScore cột tham chiếu, trọng số định danh, "đạt" = đủ mọi must_have; render `caption`; Analysis alias dài che alias ngắn; Filter bỏ thực thể w<0,6; spec bỏ must_not nhắc lại must_have | đã chạy, số liệu trong research-log |
| v1.7.2 | `59c56af`, `5192539` | một model nền = một hệ thống: Rank / loop / ảnh cuối chạy riêng theo checkpoint (`per_model`), nhãn nhóm = khoá model nền không LoRA/IP-Adapter | chạy 6/8 prompt rồi dừng chủ động (p050, C037 chưa) |
| v1.8 | `56aa202` … `3130d29` | họ FLUX.1-dev và SD 3.5 Medium; nhóm M mới; KB tự sinh trong Grounding (`kb_mode: auto`); bộ prompt S/C của nhóm | smoke 2026-09-16: SD 3.5 và FLUX **sinh ảnh được** (SD3.5 6 ảnh/45 s, 21 GB; FLUX 6 ảnh/100 s, 46 GB) |
| v1.8.1 (16/9 sáng) | `a1f2615` … `7108404` | KB tự sinh **hai bước** (chép câu nguyên văn → thuộc tính trỏ chỉ số câu), lọc phi thị giác, `analogy_en`; spec đồng bộ thẳng từ Entity tự sinh; Reflector nấc `inpaint` (sửa cục bộ OWL-ViT + inpainting) và nấc `rewrite` (VLM viết lại prompt, Idea2Img); bộ nhớ liên prompt `kb_auto/<eid>.fixes.json`; CLIP attr không None khi thiếu must_not; config vast: **Qwen2.5-VL-7B**, bỏ BLIP-2 ITM và PickScore | code xong, test offline đạt; inpaint/rewrite/7B **chưa chạy thật** |

Run trên máy vast: `runs/v17` (4 prompt gốc), `runs/v17_complex` (C002, C003, C008, C037 bản cũ), `runs/v18_smoke` (S001 mới bắt đầu).
Các lượt v1.7.1 và v1.7.2 **ghi đè** `runs/v17`; bản đã copy về máy local: `~/Research/VnCultureGen/result_kaggle/vast_v1.7/`
(`bare_vs_system_v171.html`, `bare_vs_system_final.html`, `bare_vs_system_complex_final.html`, `progress_report_final.html`).
Từ v1.8 mỗi phiên bản dùng run-name riêng (`v18`, `v18_complex`).

## 2. Kết quả chính đã có (v1.7.1, 4 model nền cũ)

- Prompt gốc (p001 áo dài, p012 thuyền thúng, p031 áo tứ thân, p050 Tết): system > bare ở 15/16 cặp; Reviewer TB bare→system:
  SD1.5 0,17→0,17, DreamShaper 0,13→0,33, SDXL 0,11→0,37, RealVis 0,08→0,30. ITM attr tăng 3/4 (SD1.5 không).
- Complex (C002, C003, C008, C037 bản cũ): 12/16 cặp; chỉ DreamShaper và SDXL tăng rõ; RealVis bằng; SD1.5 kém nhẹ.
- Kết luận tạm: hệ thống bổ trợ nhưng không vượt năng lực model nền; lợi thế co lại ngoài KB tay (thiên lệch KB) → lý do làm KB tự sinh và
  thêm SD 3.5 / FLUX.
- Loop cải thiện thật ở p050 (vòng 2) và C002 (vòng 2); 6/8 prompt dừng sau 2 vòng không tăng.
- PickScore chấm ảnh bare đẹp hơn (0,83 so với 0,28) và VQAScore bão hoà ~0,89 kể cả ảnh sai → chỉ làm cột tham chiếu.
- p031 áo tứ thân: không model nào vẽ được bốn tà; ca cho LoRA tự huấn luyện.

## 3. Lỗi đã gặp và đã sửa (để không lặp lại)

| lỗi | biểu hiện | sửa |
|---|---|---|
| top-k Reviewer theo ensemble | PickScore đẩy 5/8 ảnh bare vào top-k, hàng system không được chấm | Reviewer hai tầng: VLM chấm mọi ảnh, metric chỉ xếp trong tập đã qua |
| mốc cải thiện = top-1 Rank | vòng 1 "tốt hơn" giả (p012) | mốc = điểm Reviewer cao nhất trong pool |
| Reflector không leo nấc | 3 vòng cùng `attr_refs` | xét lần gần nhất của cách sửa; thang: ground_refs → attr_refs → more_refs → seed → guidance |
| `adapt_spec` reset `iteration=0` | mọi vòng sinh lại cùng seed (ẩn từ v1.4) | giữ iteration |
| ảnh cuối rơi vào hàng bare | p031, p050 | bare chỉ để so, không vào Rank/loop/pool |
| pool gom mọi model | ensemble giữa model, trái mục tiêu 1 model | loop riêng theo checkpoint (`per_model`) |
| web thêm must_not "conical shape" cho nón lá | VQA loại mọi ảnh nón lá đúng | spec bỏ must_not chỉ nhắc lại must_have; KB áo tứ thân bỏ "đi với nón quai thao" |
| "Tết Trung Thu" kéo theo Tết Nguyên Đán | C037 đòi quất, lì xì | alias dài che alias ngắn |
| áo dài w=0,55 lọt Filter ở C008 | đòi cổ đứng trong prompt áo tứ thân | Filter chỉ thực thể w≥0,6 |
| Reviewer loại hết → ảnh cuối None | p031 | lấy ảnh ít sai nhất làm mốc cho loop |
| must_not "wide brim" theo VQA 0,78 trên nón lá đúng | loại nhầm | must_not chỉ theo VQA cần ≥0,85 |
| `pkill -f`/`pgrep -f` giết chính phiên ssh | mất output 3 lần | dùng `[r]un_` hoặc ghép chuỗi pattern |
| nhãn nhóm = hàng có điểm cao nhất (`sdxl_refplus`) | cột "ảnh cuối loop" không khớp bảng bare/system | nhãn = khoá model nền không LoRA/IP-Adapter |
| chèn code bằng `replace(anchor, new+anchor)` | dòng `def` nhân đôi → SyntaxError (2 lần) | kiểm `ast.parse` mọi module trước commit |

## 3b. Lỗi tìm thấy khi smoke v1.8 (2026-09-16) và đã sửa

| lỗi | biểu hiện | sửa |
|---|---|---|
| Qwen 3B dựng KB một bước | thuộc tính "Is white", "Vietnamese traditional dress"; câu gốc bịa ngắn qua kiểm mờ 70%; chép ví dụ nón lá vào must_not áo dài | hai bước (chép nguyên văn câu mô tả → thuộc tính trỏ chỉ số câu), kiểm 0,8 và ≥ 8 từ, lọc từ phi thị giác, ví dụ chỉ là định dạng |
| xoá nhầm item `kb_auto` cùng item KB tay | S012: spec 0 thuộc tính, Filter "0/0", mọi ảnh "đạt", loop 0 vòng | chỉ xoá `kb@`/`kb.notes`; spec lấy thuộc tính thẳng từ Entity tự sinh (`sync_auto_entities`) |
| cổng `extract_evidence` chặn agent chỉ có `draft_kb_entry` | test fail | cổng = có một trong hai |
| bản KB tự sinh không có must_not | CLIP attr None ở FLUX, best-of-N không dừng | tương phản với confusable/câu trần; hỏi thêm must_not từ thứ dễ nhầm |
| đĩa đầy 100% khi tải Qwen 7B | tải lỗi "No space left" | xoá BLIP-2, PickScore, SD1.5, DreamShaper, zip Drive; còn ~8 GB; xoá Qwen 3B (7 GB) sau khi smoke xong |
| `pgrep -f snapshot_download` / `run_walkthrough` giết chính ssh | mất output | luôn dùng `[n]` trong pattern |

Kết quả tạm smoke (Qwen 3B, hand fallback cho áo dài): S001 realvis loop 3 vòng cải thiện ở vòng 1; S012 không hợp lệ (lỗi trên);
S021 realvis 3 vòng, sd35 2 vòng không tăng, một lỗi JSON của VLM khi mô tả ảnh. FLUX S001: bare/system/+ref đều sinh được
(6 ảnh ~100 s, đỉnh 46–47,5 GB; IP-Adapter XLabs chạy, bước tính trước embedding lùi về mã hoá từng ảnh). **Quan sát quan trọng:**
ảnh FLUX *bare* của áo dài đã đúng (cổ đứng, tà dài), còn ảnh cuối hệ thống chọn từ vòng 2 (+ref) lại KÉM hơn (tay ngắn, yếm ngực lạ)
→ với model nền mạnh, loop có thể làm xấu đi; Reviewer 3B chấm chưa tin được. Cần xem lại với 7B trước khi kết luận H20 cho FLUX.
Ảnh so sánh: `~/Research/VnCultureGen/result_kaggle/vast_v1.8_smoke/S001_montage.jpg` (bare FLUX | cuối FLUX iter2 | SD3.5 system).
S012 thuyền thúng (`S012_montage.jpg`: FLUX bare | FLUX +ref | SD3.5 bare | SD3.5 system): **FLUX bare đã vẽ thúng tròn đúng**, +ref
cũng đúng (auto_ref mở vì prior 0,10); **SD 3.5 bare ra ghe mũi nhọn, system (chỉ prompt, chưa có IP-Adapter cho SD3.5) vẫn ra ghe gỗ**
→ với SD3.5 kênh ảnh là thứ còn thiếu; với FLUX thực thể hiếm này đã có trong model. Gợi ý H20 sẽ phân hoá theo model nền:
hệ thống giúp SDXL/RealVis/SD3.5, ít hoặc không giúp FLUX ở áo dài/thúng.
Smoke **Qwen 7B** (`v18_smoke7b`, bắt đầu 07:37, 3 prompt × 7 hàng RealVis/SD3.5/FLUX): KB tự sinh áo dài của 7B =
must_have "split tunic with side slits at the hips", "two straight panels front and back", "sleeves fitting closely to the arms" (đủ 3 câu gốc
Wikipedia), must_not "left panel longer than right panel" (hiểu sai câu "tà trước ngắn hơn tà sau"), tags còn từ chung ("tradition",
"formalwear"), analogy "a long fitted tunic split into two panels, worn over wide trousers". Tốt hơn 3B rõ, còn thiếu "cổ đứng" và
must_not còn yếu. Reviewer 7B cho 28/28 ảnh qua tầng 1 (kể cả bare) → thuộc tính tự sinh hiện dễ đạt, cần xem lại tính phân biệt.
Inpaint lần đầu: OWL-ViT không tìm được "the sleeves of a Ao dai (Vietnamese long dress)" → đã sửa câu hỏi (bỏ ngoặc, dự phòng
"a sleeve" → "a Ao dai", ngưỡng thấp hơn) ở `4a90ce5`; tiến trình đang chạy vẫn dùng code cũ nên inpaint chỉ chạy thật ở lượt sau.
S012 với 7B: KB tự sinh thúng chỉ 2 must_have và một là **"covered with cow dung"** (đúng Wikipedia nhưng không nhìn thấy từ xa) → loop
SD3.5/FLUX đuổi theo thuộc tính này vô ích. Sửa `9f8769e`: bước 2 chấm `salience` 1–5 (ưu tiên hình dáng/bộ phận lớn), bỏ mục < 3 khi còn
≥ 2 mục tốt, lọc từ lớp phủ/hoá chất (dung, resin, tar, coating...). Lượt `v18_smoke7b2` (xếp hàng) chạy lại với sửa này.
S021 Trung Thu với 7B: KB tự sinh KHÔNG đủ 2 thuộc tính có gốc → lùi về bản tay; bản tay của thực thể bối cảnh liệt kê mọi yếu tố lễ hội
(bánh nướng, múa lân, ông Địa) nên Reviewer đòi cả bánh trung thu và múa lân trong ảnh trẻ rước đèn ông sao → loop đuổi thứ prompt không
nói. Sửa: `focus_context_entities` (spec) giữ must_have của thực thể bối cảnh có từ khoá trùng prompt hoặc tên thực thể vật thể trong spec;
không trùng cái nào thì giữ 2 mục đầu và hạ trọng số 0,65.
Ảnh S021 (`S021_montage.jpg`, có nhãn): cả RealVis và FLUX, bare lẫn system, đều ra đèn ông sao và trẻ em; RealVis bare treo đèn thay
vì cầm; **system của cả hai model thêm lồng đèn tròn kiểu Trung Quốc** dù must_not có "round red Chinese lantern" → negative không đủ
sức với DiT (FLUX không có negative) và với SDXL khi prompt dài. Gợi ý: Reviewer phải bắt must_not này (VQA), và Refiner dùng inpaint xoá.
Thêm nguồn **Wikipedia tiếng Anh** cho KB tự sinh (`2d65cea`, bản EN mô tả hình dáng chi tiết hơn); bản áo dài 7B lượt v2 vẫn chỉ 2
must_have ("split skirt at the sides", "fitting sleeves") → KB tự sinh còn dao động giữa các lần gọi, chưa có "cổ đứng".
**Lỗi lớn tìm được ở lượt v2**: Reviewer 7B báo THIẾU cả hai thuộc tính đó trên MỌI ảnh của cả ba model (kể cả ảnh áo dài đúng)
→ loop chạy 2 vòng vô ích rồi dừng ở cả 3 model. Nguyên nhân: thuộc tính viết theo văn Wikipedia, VLM không xác nhận được trên ảnh.
Sửa (`b0…`, xem commit "kiểm KB bằng ảnh thật"): thêm bước **`Session.validate_kb`** sau Grounding — hỏi VQA từng must_have trên
2–3 **ảnh THẬT** của thực thể (kho ảnh nhóm, đã qua CLIP); thuộc tính mà chính ảnh đúng cũng không xác nhận (< 50% số ảnh) thì bỏ
khỏi bản ghi; must_not mà ảnh đúng cũng "có" thì bỏ. Ghi `_meta.validated` (điểm từng thuộc tính) vào cache, chạy một lần mỗi thực thể.
Đây cũng là câu trả lời cho "KB tự sinh có đáng tin không": mọi thuộc tính vào Reviewer đều đã được ảnh thật xác nhận.
Lượt `v18_smoke7b3` (08:50): bước kiểm chạy 2 giây, bỏ đúng "split skirt at the sides…", Reviewer loại 3/24 ảnh (trước đó 0/24) →
bare/system bắt đầu phân biệt được. Nhưng KB áo dài còn **1 must_have duy nhất** và nó mơ hồ ("fitting sleeves, either loose or
reaching past the wrist"). Sửa tiếp: bước 2 xin **5-8** thuộc tính (lọc sau bằng ảnh thật), loại cách viết mơ hồ (either/or,
sometimes, usually…), và khi sau kiểm còn < 3 thì **lấy thêm từ bản tay nhưng cũng phải qua kiểm ảnh thật** (ghi `_meta.validated.from_hand`).

## 4. Lỗi/rủi ro còn mở

- **KB tự sinh với Qwen 3B vẫn yếu ở thực thể bối cảnh** (Trung Thu: "gather under the moonlight"); áo dài ra 2 thuộc tính đúng.
  Kỳ vọng 7B khá hơn (Culture-TRIP dùng 70B) — cần chạy lại smoke với 7B để so. Mỗi thực thể lần đầu tốn 5 lượt gọi LLM (4 nguồn + 1).
- **FLUX**: bare/system chạy; `flux_dev+ref` (IP-Adapter XLabs) chưa xác nhận; hires tắt cho họ DiT. Đỉnh 46 GB nên chỉ 1 luồng
  FLUX song song với 1 luồng khác.
- **Inpaint và rewrite chưa chạy thật** (chỉ test offline). Inpaint dùng `AutoPipelineForInpainting.from_pipe`, strength 0,85, mặt nạ hộp
  OWL-ViT nở 18%; nếu OWL-ViT không tìm vùng thì bỏ nấc.
- **Đĩa 100 GB quá chật** cho FLUX + SD3.5 + SDXL + RealVis + Qwen 7B (~81 GB cache): lần thuê sau lấy ≥ 150 GB.
- **Prompt caption**: thua legacy ở p001, thắng ở C002/C003 → chưa chốt; hàng `realvis_xl#caption` vẫn trong config.
- **Reviewer ở prompt complex**: ITM attr ≈ 0, CLIP attr ≈ 1,0 → chỉ điểm Reviewer dùng được; cần đánh giá người theo cặp.
- **Sinh và chấm cùng nguồn thuộc tính** (KB hoặc bản tự sinh): hệ thống được cộng điểm vì vẽ đúng checklist của mình. Hướng: Reviewer
  chấm theo brief Wikipedia độc lập với prompt; tập kiểm ngoài KB; đánh giá người.
- **`best_model` lệch ảnh cuối** trong một vài hồ sơ cũ (nhãn model của top-1 Rank thay vì của ảnh cuối) — đã sửa ở v1.7.2.
- **Bộ complex đổi câu** ở C002, C003, C008 (nhóm sửa 2026-09-15) → số v1.7 complex không so trực tiếp được với v1.8.

## 5. Máy vast.ai

1× A100 80 GB (Taiwan), đĩa 100 GB. Repo `/workspace/ctig17` (git pull), venv `/venv/main`, `HF_HOME=/workspace/.hf_home`,
`HF_TOKEN` trong `~/.bashrc` (bashrc return sớm ở shell không tương tác → `export HF_TOKEN=$(grep "^export HF_TOKEN" ~/.bashrc | cut -d= -f2)`).
Kho ảnh `/workspace/refs` (1.399), index `/workspace/runs/_cache/ref_index.npz`. Log `/workspace/logs/`. Script phụ `/workspace/{summ.py,
summ_pairs.py,verd.py,compare_pairs.py}`. Chạy nền bằng `setsid nohup ... &`. Host/port SSH đổi sau mỗi Start; Stop giữ đĩa, Destroy mất.

## 6. Việc kế tiếp theo thứ tự

1. Chờ smoke FLUX xong → xoá cache Qwen 3B → xoá `runs/_cache/kb_auto/*` (bản 3B) → smoke lại S001, S012, S021 với **Qwen 7B**
   (KB tự sinh, inpaint, rewrite lần đầu chạy thật) → so bản KB 7B với 3B và bản tay.
2. Chạy v1.8: S001, S012, S031, S023, S021 + C002, C003, C008, C037 (bản mới), hai luồng song song (FLUX chỉ ở một luồng), run `v18` /
   `v18_complex` → `scripts/kb_auto_export.py` cho nhóm duyệt.
3. `scripts/compare_pairs.py` cho hai run; cập nhật research-log, research-state (H20, H21, thêm H22 "negative có cần không" từ FLUX).
4. Sau đó: dev set rộng hơn (3 luồng), LoRA áo tứ thân, đánh giá người theo cặp, tách nguồn chấm khỏi nguồn sinh.
