# CTIG - Cultural Text-to-Image (Việt Nam): tóm tắt tiến độ

*Nhóm SOICT'26 · 12/09/2026 · mã nguồn github.com/OxyzGiaHuy/CTIG (v1.3.1) · chạy trên Kaggle 2×T4*

## 1. Current pipeline: what each block uses

```
Prompt (VI) ─► [1] Analysis ─► [2] Search ─► [2b] Evidence extraction ─► [3] Cultural spec ─► [4] Multi-model generation ─► [5] Scoring ─► (Review loop, disabled)
```

| block | what it uses | output |
|---|---|---|
| 1. Analysis agent | Qwen2.5-VL-3B-Instruct (local, GPU 0); every LLM call cached on disk | surface and inferred keywords, candidate entities (≤ 6), English prompt |
| 2. Search | Vietnamese Wikipedia API, DuckDuckGo text and image search VI/EN (keyless), Wikimedia Commons; full-page fetch of the top 3 pages; web cache | entity-level text evidence, prompt-level reference images, side-by-side comparison of the two query strategies |
| 2b. Evidence extraction | Qwen reads the retrieved text and returns must_have / must_not / confusable_with, each attribute with its source quote; junk filter; cross-check against a hand-written knowledge base of 38 entities | visual attributes with provenance |
| 3. Cultural spec → generation prompt | rule-based merge and ranking; must_have (EN) into the positive prompt, must_not (EN) and confusable names into the negative prompt; compel embedding concatenation when the prompt exceeds 75 CLIP tokens | GenSpec: prompt, negative prompt, seed, size, steps, guidance |
| 4. Generation | diffusers, models loaded sequentially, 4 candidates per model (best-of-N), DPM++ 2M Karras, hires fix ×1.5 (img2img, strength 0.3); áo dài LoRA (Civitai) and IP-Adapter Plus conditioned on reference images from Search | 28 images at 1024/1536 px for 7 models |
| 5. Scoring | CLIP ViT-B/32 for identity and **attribute contrast** (must_have vs must_not sentences); BLIP-2 ITM; PickScore (human-preference aesthetic model) | score table, best candidate selection, comparison grid |
| Review loop (agentic critique) | Qwen answers a closed-question checklist, revision plan, regenerate | **disabled**: the 3B VLM answers "yes" to every question; to be redesigned once the base images are solid |
| Reporting | self-contained HTML with native-resolution images, vector tables and charts; `grid_hires.png` | progress report for the advisor |

Three cache layers (LLM calls, web results, per-step artifacts) plus the generated images are persisted in `runs_cache.zip`, so re-runs cost no API calls or GPU time.

## 2. Các model sinh ảnh đã thử và kết quả (lần gần nhất: p001, v1.3, 12/09)

Prompt p001 *"Một cô gái mặc áo dài trắng đứng trước cổng trường"*, cùng seed và cùng hợp đồng văn hoá cho mọi model,
4 ứng viên mỗi model. Thước đo **CLIP attr** = xác suất CLIP dành cho "áo dài với must_have" (quần dài ống rộng, cổ
đứng, hai tà xẻ hông) so với "với must_not" (váy liền không quần, obi, cổ chéo). Danh tính CLIP và BLIP-2 ITM đạt
0,95 đến 1,00 ở mọi ảnh nên không dùng để xếp hạng.

| model | kiến trúc | attr trung bình | attr tốt nhất | ITM attr | giây / 4 ảnh | VRAM đỉnh | nhận xét bằng mắt |
|---|---|---|---|---|---|---|---|
| dreamshaper8 | SD 1.5, 512 px | 0,53 | 0,67 | 0,92 | 52 | 2,6 GB | 2/4 thành vest trắng + quần (negative quá tay) |
| sdxl_base | SDXL 1.0 | 0,56 | 0,61 | 0,86 | 230 | 13,4 GB | có quần nhưng nghiêng "áo khoác dài" |
| **realvis_xl** | RealVisXL V4 (SDXL fine-tune ảnh thực) | **0,75** | 0,80 | 0,96 | 230 | 13,4 GB | 4/4 áo dài đúng có quần |
| sdxl_aodai | SDXL + LoRA áo dài | 0,59 | 0,70 | 0,90 | 261 | 13,4 GB | áo dài ôm, tà bay tự nhiên |
| **realvis_aodai** | RealVisXL + LoRA áo dài | 0,60 | **0,86** | 0,96 | 255 | 13,4 GB | ảnh tốt nhất toàn grid (ứng viên 3) |
| sdxl_refplus | RealVisXL + IP-Adapter Plus, 3 ảnh tham chiếu | 0,80 | 0,89 | 0,74 | 217 | 9,2 GB | trang phục đúng nhưng 3/4 ảnh có 3-4 người do ảnh tham chiếu là ảnh nhóm |
| playground25 | Playground v2.5 | 0,16 | 0,21 | 0,87 | 239 | 7,2 GB | bạc màu, mờ sương: lỗi thay VAE của chúng tôi, đã hoàn lại |

Đã thử ở các bản trước và bỏ khỏi cấu hình mặc định: `sdxl_turbo` (4 bước, không dùng được negative, attr 0,43),
`sdxl_ref` (IP-Adapter bản thường, một ảnh). Đã khai báo nhưng chưa chạy: SD3.5 Medium (gated, T4 không có bf16),
HunyuanDiT (đối chứng "kéo về Trung Quốc").

![Grid p001 v1.3](report_assets/v13_grid.jpg)

*Hình 1. p001 v1.3: 7 model × 4 ứng viên, cùng seed. Viền xanh là ứng viên CLIP chọn.*

Kết luận từ p001 (một prompt, cần thêm p031, p050, p012 để chắc):

1. Negative theo must_not chặn được lỗi "váy liền không quần": từ 8/12 ảnh có quần (v1.2.1) lên gần 28/28.
2. RealVisXL hơn SDXL base trên cùng seed (0,75 so với 0,56); cộng LoRA áo dài cho ảnh tốt nhất.
3. Cùng model, khác seed lệch nhiều hơn khác model, nên sinh 4 ứng viên và chọn theo điểm là cách tăng chất lượng rẻ nhất.
4. IP-Adapter kéo cả bố cục ảnh tham chiếu; ảnh tham chiếu phải chọn theo độ khớp prompt, không chỉ theo thực thể.

## 3. Prompt sets: old vs new

| | old set (`prompts_vi.jsonl`) | new set (`prompts_complex.json`) |
|---|---|---|
| prompts | 50 | 45 |
| entities per prompt | 1 (35/50), max 4 | mostly 2 to 3, max 5; 118 mentions, 117 distinct entities |
| average length | 14 words | 25 words, with scene layout (position, time of day, action) |
| difficulty easy / medium / hard | 6 / 17 / 27 | 17 / 19 / 9 |
| taxonomy | 8 Vietnamese categories, one per prompt | 8 English categories (Everyday Life & Trades 18, Clothing 13, Performing Arts & Music 13, Architecture & Landmark 11, Landscape 9, Food & Drink 8, Festival 7, Customs & Rituals 5), multi-label |
| entity labels | knowledge-base ids (`ao_dai`) | free-text names (`"đàn nguyệt"`), not yet mapped; 47/118 mentions covered by the current knowledge base |
| reference images | 10 per prompt, 500 total | 20 per prompt, 900 total |
| failure modes measured | entity substitution | plus entity omission and cross-cultural mixing within one image |

Old-set example: *"Một cô gái mặc áo dài trắng đứng trước cổng trường."* (A young woman in a white áo dài standing at a
school gate.) New-set example (C008): *"Trong hội Lim ngày xuân, các liền anh mặc áo the khăn xếp và các liền chị mặc áo
tứ thân, đội nón quai thao đứng hát quan họ đối đáp trên chiếc thuyền rồng sơn son thếp vàng giữa hồ nước."* (At the
Lim festival in spring, men in áo the and khăn xếp and women in áo tứ thân with nón quai thao sing quan họ call-and-
response on a red-lacquered, gilded dragon boat in the middle of a lake.)

![C008: 20 reference images](report_assets/C008_refs.jpg)

*Figure 2. The 20 reference images for C008 retrieved by the team via image-search API: strong entity match, but no
single image contains the whole scene.*

Remaining work on the new set: map entities to knowledge-base ids and separate object entities from context entities
(semi-automatic with the Analysis agent, then human review); extend the knowledge base for the 70 uncovered entities
through automatic evidence extraction; filter reference images (drop AI-processed images, record source URLs, prefer
single-subject photos); rebalance difficulty, since the new set skews easy.

## Kế tiếp

Chạy p031, p050, p012 với cấu hình v1.3.1 để kiểm các kết luận trên thực thể prior thấp; xuất báo cáo HTML ảnh gốc;
bắt đầu chuẩn hoá 45 prompt mới; thiết kế lại review loop sau khi ảnh nền ổn.
