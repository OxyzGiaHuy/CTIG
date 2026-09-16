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

**Số đo S001 lượt v3** (KB chỉ còn 1 thuộc tính nên điểm Reviewer = 0 cho mọi ảnh, nhưng CLIP attr vẫn phân biệt):

| model nền | CLIP attr bare | CLIP attr system | hạng ensemble bare → system |
|---|---|---|---|
| RealVis XL | 0,92 | 0,95 | 0,68 → 0,90 |
| SD 3.5 Medium | 0,82 | 0,90 | 0,37 → 0,54 |
| FLUX.1-dev | 0,74 | 0,86 | 0,21 → 0,40 |

→ H20 được ủng hộ trên cả ba model kể cả FLUX, theo CLIP attr. Reviewer chưa dùng được cho tới khi KB đủ thuộc tính.
Đang chạy `v18_smoke7b4` với KB đầy đủ (5–8 thuộc tính, lọc bằng ảnh thật, bổ sung bản tay có kiểm) + trang so sánh cuối lượt.

**Hai lỗi nữa tìm thấy khi đọc điểm kiểm KB của v4 (đã sửa, `f5750f6` + `81afa6c`):**
1. *Câu hỏi VQA sai ngữ pháp.* Ghép "Does the ao dai have **worn over wide-legged trousers**?" → Qwen trả No cho cả ảnh áo dài
   thật (cổ đứng 0,33; quần ống rộng 0,00). Đổi sang dạng phát biểu: `Look carefully at the {name}… Statement: "{attr}". Is this
   statement true?`. Đo lại trên ảnh thật: cổ đứng 0,91 → 0,93; tà xẻ 0,56 → 0,78; quần ống rộng 0,22 → 0,38. Dùng chung cho
   **Reviewer** (ảnh hưởng mọi số Filter trước đây) và bước kiểm KB.
2. *Kho ảnh có bản trùng.* `index_refs` trả 3 ảnh nhưng cùng một file (kho có ảnh giống nhau ở `evidence_images` và
   `evidence_images_complex`) → "kiểm bằng 3 ảnh thật" thực chất chỉ 1 ảnh. Khử trùng theo (tên, kích thước) khi đánh chỉ mục và
   theo tên khi chọn ảnh kiểm. **Cần dựng lại `ref_index.npz`** trước lượt tiếp theo.

## 3c. Bộ dữ liệu mới của nhóm (2026-09-16, bản thứ hai)

Drive `1R2C8iw3ruOE4UFRtnpaYWb193OKCYHnI` (tải bằng `gdown --folder`): prompts_simple/complex (bản này chỉ đổi **S046**: "làng chài,
thuyền thúng" → "cầu khỉ", để không trùng entity với S012) và **ảnh tham chiếu theo PROMPT**:

```
reference_images_simple/selected/<S###>/01.jpg,02.jpg,03.jpg   (3 ảnh chọn tay, 150 ảnh)
reference_images_simple/candidates/<S###>/…                    (~20 ảnh/prompt, 1000 ảnh)
reference_images_complex/selected|candidates/<C###>/…          (196 + 999 ảnh)
```

Đã nối vào pipeline (`retrieval.ref_dir`, `ref_dir_candidates`): `Session.prompt_refs()` là **tầng ưu tiên nhất** cho IP-Adapter
(không cần CLIP tìm trong kho trộn, không chọn nhầm) và là ảnh thật cho bước kiểm KB. Kho cũ 1.399 ảnh (`/workspace/refs`) vẫn là
đường lùi khi prompt không có thư mục riêng. **Lưu ý báo cáo:** ảnh này do nhóm chọn tay → phải khai báo như KB tay; muốn đo "hoàn toàn
tự động" thì đặt `ref_dir_candidates: true` (dùng ảnh chưa lọc) hoặc tắt `ref_dir`.

## 3d. Kết quả lượt v4 (`v18_smoke7b4`, xong 09:54) — 3 prompt × 3 model nền, Qwen 7B

Trung bình S001, S012, S021 (KB tự sinh có kiểm ảnh thật, chưa có ảnh ref theo prompt, VQA còn dạng câu hỏi cũ):

| model nền | CLIP attr bare → system | điểm Reviewer bare → system |
|---|---|---|
| RealVis XL | 0,77 → 0,87 | +0,30 → +0,40 |
| SD 3.5 Medium | 0,50 → 0,81 | +0,37 → +0,53 |
| FLUX.1-dev | 0,57 → 0,75 | +0,45 → +0,57 |

**System > bare ở cả ba model trên cả hai số đo** — lần đầu điểm Reviewer cũng phân biệt được (trước đó mọi ảnh cùng 0).
Loop có tác dụng: S001 đạt ở vòng 2, S012 lấy ảnh vòng 1, S021 dừng vì không cải thiện. Ảnh cuối của S001 và S012 đều từ hàng
`realvis_xl+ref`, tức kênh ảnh vẫn là thành phần quyết định.

Lượt **v5** đang chạy (`v18_v5`, 4 prompt S001/S012/S021/S031, 8 hàng gồm `flux_dev+ref`): thêm ảnh ref theo prompt của nhóm,
VQA dạng phát biểu, chỉ mục kho dựng lại sau khử trùng (1.346 ảnh sau khi bỏ 53 bản trùng).

**Kiểm KB với ảnh của nhóm + VQA phát biểu cho kết quả sạch hẳn** (áo dài, S001):

| thuộc tính | điểm trên 3 ảnh thật | kết quả |
|---|---|---|
| high stand-up mandarin collar | 1,00 (trước 0,33) | giữ |
| long-sleeved tunic split at the hips… | 1,00 (trước 0,67) | giữ |
| fitted bodice with flowing loose panels | 1,00 | giữ |
| worn over wide-legged long trousers | 0,33 | bỏ (ảnh nhóm là cận cảnh nửa người) |
| split skirt…, front skirt shorter…, seam along the side | 0,00 | bỏ |

Đáng chú ý: **cả 3 thuộc tính giữ lại đều đến từ bản tay**, mọi thuộc tính LLM tự sinh cho áo dài đều bị ảnh thật loại. Với thực thể
có bản tay, tự sinh chưa bằng; giá trị thật của tự sinh là ở ~113 thực thể NGOÀI KB của hai bộ prompt mới.

## 3e. v1.8.3 — KB tự sinh làm ĐÚNG cách viết KB tay (2026-09-16 chiều)

Nhận xét của user: "đang bị bias vào KB viết tay; tại sao không dùng chính cách tạo ra KB tay để áp cho mọi prompt". Đúng — bằng chứng:
ở v5, cả 3 thuộc tính áo dài giữ lại đều `from_hand`. Nguyên nhân là ràng buộc **"mỗi thuộc tính phải kèm câu gốc chép nguyên văn"**:
nó ép LLM copy câu Wikipedia dài ("split skirt on both sides of the hips, extending from the waist to mid-thigh") thay vì viết cụm
ngắn như người ("high stand-up mandarin collar"). Người viết KB tay **nhìn ảnh** và **viết cụm ngắn**, không trích dẫn.

Cách mới (`b586f3b`), ba bước đúng như quy trình tay:
1. đọc nguồn → chép các câu mô tả hình dáng làm **tư liệu** (không còn là ràng buộc trích dẫn);
2. VLM **NHÌN 2–3 ảnh THẬT của chính prompt** (thư mục `selected/<id>` của nhóm) + tư liệu → **viết** must_have là cụm 3–6 từ, hai cụm
   đầu là đặc điểm định danh, `confusable_with` là vật dễ nhầm nhất, và must_not **viết từ vật dễ nhầm đó**; kèm tags, analogy, prior;
3. kiểm lại bằng chính ảnh thật (`validate_kb`) — thay hoàn toàn cho ràng buộc câu gốc.
Bản tay chỉ còn là **đường cứu khi tự sinh trắng** (< 2 thuộc tính), không còn là nguồn ưu tiên.

Ngoài ra `viz.final_grid`: mỗi prompt có **lưới kết luận** ở đầu walkthrough.html — mỗi model nền một hàng, trái là ảnh model thuần,
phải là ảnh cuối hệ thống, kèm Δ CLIP attr / Reviewer / VQAScore. Đã dựng lại cho v4; bản local:
`~/Research/VnCultureGen/result_kaggle/vast_v1.8_smoke/v4/S001_walkthrough.html` (và S012, S021).

## 3f. Kết quả v5 (`v18_v5`, xong 10:57) — 4 prompt (S001, S012, S021, S031) × 3 model nền, có ảnh ref của nhóm

| model nền | CLIP attr bare → system | điểm Reviewer bare → system |
|---|---|---|
| RealVis XL | 0,75 → 0,92 | +0,18 → +0,45 |
| SD 3.5 Medium | 0,49 → 0,86 | +0,31 → +0,45 |
| FLUX.1-dev | 0,59 → 0,65 | +0,30 → +0,47 |

System > bare ở cả 3 model, cả hai số đo (9/12 cặp theo tiêu chí của `compare_pairs`). Khoảng cách lớn nhất ở SD 3.5 (+0,37 CLIP attr)
và RealVis (+0,27 Reviewer); FLUX ít hưởng lợi nhất về CLIP attr (+0,06) — khớp với quan sát "model mạnh đã biết thực thể phổ biến".
Trang so sánh: `/workspace/runs/v18_v5/bare_vs_system.html`.

**Kênh ảnh theo model** (điểm khác biệt còn lại): RealVis có IP-Adapter (h94) và FLUX có IP-Adapter XLabs; **SD 3.5 Medium KHÔNG có**
(InstantX chỉ phát hành cho 3.5 Large). Đang cân nhắc thêm hàng `+init` (img2img từ ảnh tham chiếu, strength ~0,75) để mọi họ model
có cùng một dạng kênh ảnh so được với nhau; rủi ro chép ảnh đã có số đo "giống ref" (ngưỡng 0,88) bắt.

**ImageRAG (2502.09411) kết hợp ảnh thế nào** (để đối chiếu thiết kế của ta): SDXL dùng **IP-Adapter, ip_adapter_scale = 0,5, 1 khái
niệm × 1 ảnh**; OmniGen dùng in-context, tối đa 3 ảnh, 3 khái niệm × 1 ảnh. **Mỗi ảnh đi kèm caption trong prompt** theo mẫu
"According to these examples of <c1>:<img1>, …, generate <p>". Truy hồi **một vòng**, kích hoạt bằng VLM hỏi "ảnh có khớp prompt
không" rồi nêu khái niệm thiếu. CTIG hiện: scale 0,4 (thấp hơn), tối đa 3 ảnh cho IP-Adapter Plus, **chưa nối caption của ảnh vào
prompt** — nên thử scale 0,5 và thêm caption theo mẫu của họ.

## 3g. v1.9 — 8 lỗi của Reviewer/loop tìm bằng agent phân tích output thật (2026-09-16)

Agent đọc toàn bộ `runs/v18_v5` (4 prompt × 3 model, 622 câu VQA) và **xem ảnh**. Lỗi và cách sửa (commit `b418b63`, `ea8bb72`, `74d7f8e`):

| # | Lỗi | Bằng chứng | Sửa |
|---|---|---|---|
| 1 | `garment_rules` nhánh sash/belt luôn trả `absent` (không phân biệt "mô tả không nói" với "không có") | 24/32 ảnh S031 bị ghi thiếu "silk sash" dù VQA trung vị 0,94 | chỉ `absent` khi trường có mặt |
| 2 | `must_not` rỗng: `validate_draft` bỏ qua `hand.must_not_en` | S001 `forbidden_attrs: []`; 23/26 ảnh có "bare legs" trong mô tả, 0 ảnh bị phạt → **ảnh cuối không quần vẫn +1,00** | luôn nạp must_not tay |
| 3 | Bỏ phiếu 3 ảnh quá nhiễu | cùng 3 ảnh, 2 lần chạy: "mandarin collar" 1,00 và 0,33 | khoá 2 thuộc tính định danh, chỉ bỏ khi ảnh BÁC BỎ rõ |
| 4 | Ngưỡng cứng 0,75 / 0,85 | 0,75 cắt giữa hai mode 0,731 (14 lần) và 0,755 (11 lần); must_not 0,85 bỏ sót 25 câu trong 0,60–0,85 | 0,60 / 0,70 + **điểm phần** cho dải 0,35–0,60 |
| 5 | Thuộc tính "chết" | S012 "on a central Vietnam beach" max 0,04/28 ảnh; S021 "lion dance" max 0,22/36 | max VQA < 0,5 trên mọi ảnh → bỏ khỏi điểm và khỏi mục tiêu loop |
| 6 | Hoà điểm hàng loạt | S001 16/26 ảnh cùng +1,00; S021 23/36 cùng +0,20 | khoá phá hoà: Reviewer → VQA TB → VQAScore → ensemble |
| 7 | Rank rỗng → ảnh cuối None | | lấy ảnh hệ thống ít sai nhất |
| 8 | must_have của thực thể phụ không có trong prompt | S021 "lion dance" trọng số 2/5 → trần điểm 0,60 | `focus_context_entities` áp cho cả vật thể phụ |

**Giả thuyết của tôi trước đó SAI**: ảnh tham chiếu S001 là ảnh **toàn thân**, thấy rõ quần; thuộc tính quần rớt vì VQA đọc sai
áo trắng trên quần trắng, không phải vì khung ảnh.

**Định vị vùng cho inpaint** (theo SLD / GenArtist / Marmot, agent nghiên cứu riêng): không hệ thống nào định vị thuộc tính trừu
tượng; tất cả định vị **đối tượng** rồi suy ra vùng. Đã đổi theo: (1) **Qwen2.5-VL grounding** trả bbox (toạ độ theo ảnh đã resize,
quy đổi `image_grid_thw * 14`), (2) OWL-ViT với **danh từ ngắn** (`collar`, `brim` — GenArtist dùng từ vựng 7.605 danh từ đơn),
(3) box **thực thể cha** (SLD). Thuộc tính chuyển sang **prompt của vùng inpaint** (DiffEdit) thay vì câu truy vấn.

Trạng thái 5 khuyến nghị của agent nghiên cứu (commit `ea8bb72` + `151c977`):

| # | khuyến nghị | trạng thái |
|---|---|---|
| 1 | Qwen2.5-VL grounding làm bộ định vị chính | xong, `qwen_vl.ground()` + `PromptAgent.locate()`, quy đổi `image_grid_thw*14` |
| 2 | Thủ thuật hai lượt SLD, hỏi thực thể cha, thuộc tính vào prompt inpaint | xong, nấc 4 là hộp cha; prompt vùng = `"<thực thể> with <thuộc tính>"` |
| 3 | crop-then-ground hai giai đoạn | xong, `crop_then_ground()`, crop hộp cha + phóng lên 768 px, hỏi lại bộ phận, map ngược; lệch IoU < 0,1 so với hộp toàn ảnh thì tin hộp zoom |
| 4 | detector mở từ vựng với danh từ ngắn | xong ở dạng OWL-ViT + danh từ ngắn (chưa đổi sang GroundingDINO; cân nhắc nếu OWL-ViT vẫn yếu) |
| 5 | fallback phân tầng + SAM refine | xong, thang 4 nấc; `sam_refine()` dùng `facebook/sam-vit-base`, chọn trong 3 mask whole/part/subpart theo tỉ lệ diện tích 0,15–1,2 lần hộp. Mask tương phản kiểu DiffEdit **chưa làm** (nấc 5 dự phòng) |

Cần tải thêm `facebook/sam-vit-base` (~375 MB) trên máy trước lượt v19; đĩa còn ~4,9 GB.

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
