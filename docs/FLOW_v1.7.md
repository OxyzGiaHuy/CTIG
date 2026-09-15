# CTIG v1.7.1 — Flow hệ thống và model dùng ở từng bước

Cập nhật 2026-09-15. Cấu hình tham chiếu: `configs/vast_a100.yaml` (1× A100 80 GB). Ba khối:

```
Prompt (VI) ──► GROUNDING ──► GENERATE: model nền M × {bare, system} ──► AGENTIC REVIEW LOOP ──► ảnh cuối + hồ sơ
```

Mục đích của nhánh `bare` cạnh `system` trên **cùng model nền, cùng seed**: chứng minh "M + hệ thống" đúng văn hoá hơn "M trần"
là tính chất chung của mọi model nền, không phải chọn model tốt nhất. Khi triển khai chỉ gắn một model.

---

## 1. Grounding — prompt → thực thể, thuộc tính dương/âm, brief, ảnh tham chiếu

| Bước con | Làm gì | Model / nguồn |
|---|---|---|
| Analysis | Tách keywords, chọn thực thể ứng viên từ KB, dịch `prompt_en`. Bỏ ứng viên không có căn cứ trong prompt; alias dài che alias ngắn ("Tết Trung Thu" không kéo theo Tết Nguyên Đán). | **Qwen2.5-VL-3B-Instruct** (agent văn bản, temperature 0) + luật |
| Search | Text VI/EN và ảnh cho từng thực thể và cho prompt gốc; đọc cả trang. Cache trên đĩa theo khoá băm. | Wikipedia VI API, DuckDuckGo (`ddgs`) text + ảnh, Wikimedia Commons; Serper tuỳ chọn |
| Extraction | Rút thuộc tính must_have / must_not / dễ nhầm từ văn bản web, phải kèm câu gốc. | Qwen2.5-VL-3B |
| Summary agent | Brief thị giác cho mỗi thực thể theo bốn chiều (hình dáng, chất liệu/màu, bối cảnh, khác gì với thứ dễ nhầm), câu "vẽ thế nào". | Qwen2.5-VL-3B |
| Spec | Ghép KB viết tay (38 thực thể, `data/kb/entities.json` v0.4) với thuộc tính web thành `must_have_en` / `must_not_en` / `tags_en` / `neg_tags_en`; trọng số theo thứ tự nêu tên; bỏ must_not chỉ nhắc lại must_have. | luật (RuleAgent) |
| Ảnh tham chiếu | Tầng 0: kho 1.399 ảnh của nhóm đánh chỉ mục CLIP (cosine ≥ 0,26). Tầng 1–3: ảnh web theo CLIP ≥ 0,75 → ≥ 0,50 → ảnh của prompt gốc. Filter agent bỏ ảnh nhóm khi prompt một người, bỏ ảnh thiếu must_have. Cắt theo thực thể. | **CLIP ViT-B/32** (`openai/clip-vit-base-patch32`), Qwen2.5-VL-3B (Filter), **OWL-ViT base patch32** (cắt), CLIP quét lưới khi OWL-ViT không thấy |
| GenSpec | Render prompt cho bộ sinh. `legacy` (mặc định): cảnh + tên thực thể + 3 must_have câu dài + hậu tố ảnh chụp; negative = chung + tên dễ nhầm + must_not. `bare`: `prompt_en` + negative chung. `caption`: một câu tự nhiên ≤ 75 token, 2 thuộc tính định danh. Prompt > 77 token nối embedding bằng **compel**. | luật; tokenizer CLIP để đếm token |

Đầu ra: một bảng Grounding (thực thể | dương → prompt | âm → negative | brief | ảnh tham chiếu) và `step_*.json` cho từng bước con.

---

## 2. Generate — model nền M × {bare, system}

Nhóm M hiện tại và hàng trong `models:`:

| M | Checkpoint | Hàng bare | Hàng system |
|---|---|---|---|
| SD 1.5 gốc | `stable-diffusion-v1-5/stable-diffusion-v1-5`, 512 px | `sd15_base#bare` | `sd15_base` |
| DreamShaper 8 | `Lykon/dreamshaper-8` (SD 1.5 fine-tune), 512 px | `dreamshaper8#bare` | `dreamshaper8` |
| SDXL 1.0 gốc | `stabilityai/stable-diffusion-xl-base-1.0` + VAE `madebyollin/sdxl-vae-fp16-fix`, 1024 px | `sdxl_base#bare` | `sdxl_base`, `sdxl_base+ref` |
| RealVis XL 4.0 | `SG161222/RealVisXL_V4.0` (SDXL fine-tune), 1024 px | `realvis_xl#bare` | `realvis_xl`, `realvis_xl#caption`, `realvis_xl+ref`, `realvis_aodai`, `realvis_aodai+ref`, `sdxl_refplus` |

Thành phần hệ thống bật ở nhánh system (hàng `+ref` / LoRA / refplus là ablation từng thành phần trên cùng M):

| Thành phần | Chi tiết | Model |
|---|---|---|
| Prompt/negative từ Grounding | render `legacy`, compel khi dài | như trên |
| Scheduler | DPM++ 2M Karras, 30 bước, guidance theo model (RealVis 5,0; SDXL 6,5; SD 1.5 7,0–7,5) | diffusers |
| Best-of-N thích nghi | sinh 2, verifier CLIP attr chưa đạt 0,70 thì thêm 2, tối đa 6. Bare sinh cố định 6 (bằng ngân sách tối đa của system). | CLIP ViT-B/32 |
| Hires fix | phóng ×1,5 rồi img2img strength 0,3, 20 bước; bật cả cho hàng IP-Adapter trên 80 GB | cùng pipeline |
| Ảnh tham chiếu `+ref` | IP-Adapter SDXL, scale 0,4, ảnh đã Filter và cắt; chỉ dùng khi prior của thực thể ≤ 0,45 (model không tự vẽ được) hoặc khi Refiner ép | `h94/IP-Adapter` `ip-adapter_sdxl.bin` |
| `sdxl_refplus` | IP-Adapter Plus ViT-H, scale 0,4, luôn dùng ảnh | `h94/IP-Adapter` `ip-adapter-plus_sdxl_vit-h.safetensors` |
| LoRA áo dài | chỉ khi spec có áo dài, scale 0,8, trigger `aodaixl` | Civitai "JAY - AO DAI XL" (version 590793) |

Số đo cho mỗi ảnh (bảng điểm, dùng để xếp hạng, không thay được Reviewer):

| Số đo | Cách tính | Model |
|---|---|---|
| CLIP attr | P("thực thể with must_have") so với P("with must_not") | CLIP ViT-B/32 |
| ITM attr | trung bình xác suất khớp ảnh với từng câu must_have | **BLIP-2 ITM ViT-g** (`Salesforce/blip2-itm-vit-g`) |
| Đẹp | PickScore chuẩn hoá min-max trong lần chạy | **PickScore v1** (`yuvalkirstain/PickScore_v1`) |
| Giống ref | cosine CLIP ảnh–ảnh với ảnh tham chiếu; > 0,88 coi là chép | CLIP ViT-B/32 |
| Hạng ensemble | 1 − trung bình hạng trên các verifier có (Ma et al. 2025), trừ phạt chép | — |
| VQAScore | P(Yes \| "Does this figure show "<prompt>"?") (Lin 2024); cột tham chiếu, bão hoà ~0,9 | Qwen2.5-VL-3B |

CLIP identity và ITM danh tính bị ẩn vì bão hoà 0,95–1,00 trên mọi ảnh.

---

## 3. Agentic Review Loop — Reviewer → Reflector → Refiner → chọn trên toàn pool

| Agent | Làm gì | Model |
|---|---|---|
| **Reviewer, tầng 1 (Filter)** | Với MỌI ảnh của MỌI hàng (kể cả bare, để lập bảng bare/system): VLM *mô tả* ảnh (số người, trang phục, vật, nền) → agent văn bản khớp mô tả với must_have/must_not, phải trích được cụm → luật trang phục → hỏi có/không trên ảnh cho từng thuộc tính ("Does the ao dai in this photo have a high stand-up mandarin collar?"): ≥ 0,75 xác nhận must_have, ≥ 0,85 thêm must_not, ≤ 0,25 bác → CLIP phủ quyết must_not khi VLM đọc nhầm. Điểm = (Σw có − Σw must_not)/Σw, w = 2 cho hai thuộc tính định danh đầu của KB. Ảnh có must_not hoặc sai số người bị loại. | Qwen2.5-VL-3B (mô tả, khớp chữ, VQA), CLIP ViT-B/32 (phủ quyết) |
| **Reviewer, tầng 2 (Rank)** | **Riêng cho từng model nền** (một checkpoint = một hệ thống; các hàng +ref / LoRA / IP-Adapter Plus của cùng checkpoint thuộc một nhóm, không ensemble giữa các model): trong ảnh system của nhóm đã qua tầng 1, xếp theo (điểm Reviewer, hạng ensemble) lấy top-k = 8; LLM xếp hạng hai lượt **đảo thứ tự trình bày** rồi trung bình hạng, đối chiếu metric bằng Spearman. Hàng bare không vào Rank, loop hay ảnh cuối. Không ảnh nào qua thì lấy ảnh ít sai nhất làm mốc. | Qwen2.5-VL-3B |
| **Reflector** | Từ chẩn đoán của ảnh mốc (điểm Reviewer cao nhất trong pool): kế hoạch sửa bằng luật (thiếu gì thêm vào prompt hoặc nhấn compel ×1,3 nếu đã có, must_not gì thêm negative, sai số người thêm "single person", còn thiếu thì kèm ảnh tham chiếu). Bộ nhớ các cách đã thử; cách vừa rồi có tăng thì giữ và đổi seed, không tăng thì **leo nấc**: ảnh Grounding → ảnh truy hồi theo caption thuộc tính thiếu (LLM viết caption, ImageRAG) → thêm ảnh và scale IP-Adapter +0,1 → seed → guidance +1,5. Dừng sau 2 vòng liền không cải thiện hoặc khi ảnh đủ mọi must_have. | luật + Qwen2.5-VL-3B (caption) |
| **Refiner** | Sinh lại trên model system tốt nhất theo kế hoạch, seed dịch theo vòng, ảnh vào `revision/iter<n>/`, tối đa 3 vòng, **giữ hết ảnh**. Truy hồi ảnh theo caption: kho CLIP trước, DuckDuckGo ảnh sau, lọc CLIP theo caption, cắt theo thực thể. | cùng pipeline sinh + IP-Adapter; CLIP; Qwen (Filter lại ảnh mới) |
| **Chọn cuối** | Trên **toàn pool của model nền đó** (ứng viên gốc + mọi vòng) theo điểm Reviewer, hoà thì ưu tiên ảnh sớm hơn (không lấy vòng cuối). Mỗi model nền có một ảnh cuối; hồ sơ chính hiển thị là `agents.primary_model`. | — |

Đầu ra: ảnh cuối, `step_candidate_review.json` (mô tả từng ảnh, VQA từng thuộc tính, kế hoạch từng vòng, pool), bảng
"Bare vs system" (Δ CLIP attr, ITM attr, hạng ensemble, Reviewer TB, VQAScore, Filter đạt), trang so sánh
`scripts/compare_pairs.py`, và `progress_report.html` cho cả run.

---

## Chi phí trên 1× A100 80 GB

Một prompt với 13 hàng: Grounding 15–300 s (tuỳ cache web), Generate 100–700 s, Review Loop 250–1.600 s (mỗi vòng ~2–4 phút, VQA
~1 s một ảnh cho ~14 câu hỏi). Hai tiến trình song song theo prompt dùng GPU 99%, VRAM ~25 GB mỗi tiến trình.

## Chưa có / kế tiếp

- SD 3.5 Medium (họ DiT, `sd35_medium` đã có trong registry, chưa IP-Adapter và hires) và AltDiffusion m18 (baseline nhận thẳng prompt tiếng Việt).
- LoRA tự huấn luyện cho thực thể model không vẽ được (áo tứ thân) từ kho ảnh của nhóm.
- Đánh giá người theo cặp bare/system trên trang so sánh.
