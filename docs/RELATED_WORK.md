# Related work & nguồn đã dùng khi hình thành SAVIER

Tài liệu tra cứu nội bộ: mỗi công trình → nó ảnh hưởng quyết định nào của method, dùng ở đâu trong bài, và tình trạng
(đã chạy / chỉ trích dẫn / không chạy). BibTeX đầy đủ ở `docs/references.bib` (trích từ bản nháp SOICT'26, đã bỏ key lặp).
Phần Related Work viết sẵn của bài: `MSLab/SOICT'26/Cultural_Image_Generation_SOICT2026/contents/2_related_work.tex`.
Mốc tra cứu: 2026-09-16 (HANDOFF §3j) và 2026-09-19 (kiểm tên metric tự đề xuất).

## 1. Nền trực tiếp của pipeline (đã chạy hoặc dùng làm thành phần)

| Công trình | Nguồn | Dùng làm gì trong SAVIER | Quyết định nó gây ra |
|---|---|---|---|
| **Culture-TRIP** — Jeong et al., NAACL 2025 | arXiv 2502.16902 · `Kakaomacao/Culture-TRIP` | Nền tinh chỉnh prompt: `data/culture_trip/<pid>.json` → P_ct → I0 (arm "refined prompt") | Điểm yếu nêu được một câu: vòng tinh chỉnh chấm **prompt**, không nhìn **ảnh** → SAVIER chèn tầng quan sát ảnh phía sau. Chạy với qwen2.5:14b thay llama3:70b (khai báo Limitations). R **không đọc** phần mở rộng Culture-TRIP làm bằng chứng vì nó nhiễu (S011 "Chinese opera", S039 "Taj Mahal", S016 "Borneo"). |
| **T2I-Copilot** — Chen et al., ICCV 2025 | arXiv 2507.20536 · MIT | Khung 3 agent một backbone (interpret / generate / evaluate) → C/O/R; ý "prompt-blind" Observer; ablation của họ (bỏ Quality Evaluator chỉ mất 0,008) là lý do ta bỏ gate và vòng lặp | 17/09: bỏ KB tay, dựng lại theo khung này (LOOP_v2.md). |
| **IP-Adapter (Plus ViT-H)** — Ye et al., 2023 | arXiv 2308.06721 · `h94/IP-Adapter` | Điều kiện ảnh tham chiếu cho SDXL; XLabs `flux-ip-adapter` cho FLUX | Thay img2img (0,35/0,60 không đổi được định danh vật) bằng cùng seed + IP-Adapter. |
| **SDXL 1.0** — Podell et al., 2023 | arXiv 2307.01952 | Generator 1 (fp16 VAE fix, DPM++ 2M Karras, compel >77 token) | |
| **FLUX.1-dev** — Black Forest Labs, 2024 | HF model card | Generator 2 (T5 512 token đọc được câu thêm — giải thích vì sao agent chỉ có gain trên FLUX) | |
| **Mistral Small 3.1 24B** — Mistral AI, 2025 | mistral.ai | Backbone duy nhất cho C/O/R | |
| **Qwen2.5-VL-7B** — Bai et al., 2025 | arXiv 2502.13923 | Evaluator độc lập (≠ backbone agent), VQAScore-variant, RRR | |
| **Wikipedia tiếng Việt** | vi.wikipedia.org | Nguồn bằng chứng cho Evidence Card (`quote_vi` nguyên văn, ≤4k ký tự cắt quanh danh từ thực thể) | Thay KB tay do Claude soạn (không có tư cách làm chuẩn). |

## 2. Metric (đã chạy)

| Công trình | Nguồn | Dùng làm gì | Ghi chú trung thực |
|---|---|---|---|
| **VQAScore** — Lin et al., ECCV 2024 | ecva.net 1435_ECCV_2024 · GenAI-Bench | Công thức P(Yes) với câu hỏi gốc theo P0 | Backbone clip-flant5-xxl không dựng được (t2v_metrics pin torch 2.5.1/transformers 4.49) → dùng Qwen2.5-VL; so được giữa các arm, **không** so với số công bố. |
| **CAIRE** — Yayavaram et al., EACL 2026 | aclanthology 2026.eacl-long.389 | Cultural relevance VN/CN trên 300 ảnh (mã tác giả, 37 GB asset) | Đo *relevance* không đo *đúng thực thể*; 15 đơn vị I1 chạy lại chưa chấm lại. |
| **GIE-Bench / AugCLIP / EditVal** | (id arXiv chưa kiểm — tra lại trước khi trích) | Ý tách "sửa được" vs "giữ được" cho CALR/SIR/RE/GDF1-proxy | Sáu tên metric tự đặt chưa ai dùng; **viện dẫn ý, không claim**. |
| **FAGER** — Lim et al., CVPR-W 2026 | arXiv 2605.19111 · MIT | Mốc A/B thật-vs-sinh 0,97 (ta 0,58 → thước đo mù, dẫn tới sửa bộ chấm 16/09); rubric sự kiện có thật → VQA | Là tiền lệ thật của "visual facts + refine"; chưa chạy làm thước đo chéo. |

## 3. Tiền lệ gần nhất (phải nhắc trong Related Work, không claim vượt)

| Công trình | Nguồn | Khác gì SAVIER | Tình trạng |
|---|---|---|---|
| **ImageRAG** — Shalev-Arkushin et al., ICLR 2026 | arXiv 2502.09411 · `rotem-shalev/ImageRAG` (không LICENSE) | Cũng: VLM so ảnh nháp với yêu cầu → tìm ảnh tham chiếu → sinh lại, đúng SDXL + h94/IP-Adapter. Khác: không bằng chứng văn bản có trích dẫn, không tách preservation/identity. | Đã clone `/workspace/baselines/`, **chưa chạy so trực tiếp** (máy cũ mất). |
| **FAGER** | như trên | Rubric từ ảnh tham chiếu + LLM; có subset văn hoá | Trích dẫn; chưa so. |
| **AHEaD / CULTIVate** — ICLR 2026 | arXiv 2511.05681 (không LICENSE) | Hiệu chỉnh ngưỡng bộ kiểm VLM trên ảnh thật (ý ta tưởng mới) | Trích dẫn. |
| **Idea2Img** — Yang et al., ECCV 2024 | (mã chết 2024) | Phản hồi từ ảnh ứng viên → sửa prompt | Trích dẫn. |
| **Self-correcting LLM-controlled Diffusion (SLD)** — Wu et al., CVPR 2024 | Apache | Sửa vật/bố cục qua hộp; ý "hộp thực thể cha" từng dùng cho inpaint (đã bỏ) | Trích dẫn. |
| **GenArtist** — Wang et al., NeurIPS 2024 | (mã hỏng) | MLLM agent sinh+sửa; từ vựng 7.605 danh từ cho OWL-ViT (đã bỏ) | Trích dẫn. |
| **Marmot** — Sun et al., 2025 | arXiv 2504.20054 | Multi-agent sửa theo vật | Trích dẫn. |
| **Generation Navigator** — 2026 · **TARA** ("One Rewrite to Fix Them All?") — 2026 | arXiv 2605.17969 · 2607.18724 | Hành động phẳng / định tuyến kiểu sửa; gần nhất với "thang leo" đã bỏ | Trích dẫn. |
| **Ma et al., Inference-time scaling for diffusion** — 2025 | arXiv 2501.09732 | Best-of-N thắng vòng lặp cùng ngân sách trên FLUX → lý do đề xuất verify-and-select K=3 và lý do bỏ vòng lặp | Trích dẫn. |
| **MosAIG / When Cultures Meet** — Bhalerao et al., ACL Findings 2026 | | Agent persona văn hoá để *soạn* prompt, không quan sát ảnh | Trích dẫn. |

## 4. Bối cảnh văn hoá & benchmark (khoảng trống Việt Nam)

| Công trình | Nguồn | Vai trò trong bài |
|---|---|---|
| Where Culture Fades — Shi et al., CVPR 2026 | | Failure mode ở mức thực thể văn hoá |
| CCUB — Liu et al., 2023 · SCoFT — CVPR 2024 | arXiv 2301.12073 | Nhánh fine-tune dữ liệu văn hoá; **không có Việt Nam** |
| PEA-Diffusion (ECCV 2024) · AltDiffusion (AAAI 2024) | | Nhánh đa ngữ; AltDiffusion đã gồm tiếng Việt → không viết "chưa có benchmark Việt" |
| CultureCLIP — 2025 | arXiv 2507.06210 | Recognition ≠ repair; ý "confusable" |
| CuRe (ICCV 2025) · CulturalFrames (EMNLP Findings 2025) · Beyond Aesthetics/CUBE (NeurIPS 2024) · Culture in Action (ICLR 2026) · RusCode (NAACL Findings 2025) · Diffusion through a Global Lens (ACL 2025) | | Benchmark văn hoá; Việt Nam vắng trong CuRe/CCUB/CULTIVate → lý do đề tài |
| Image Transcreation — Khanuja et al., EMNLP 2024 | | Chuyển văn hoá ảnh |
| **ViFA-Council** — Nguyen et al., MAPR 2026 (HCMUS) | arXiv 2609.13348 (không LICENSE) | Bài sinh ảnh văn hoá Việt duy nhất có mã; chỉ gọi API, không truy hồi, không kiểm |
| **VietFashion** — Cao et al., ICMR 2026 (HCMUS) | | Từ vựng thuộc tính áo dài dùng lại được cho contract |

## 5. Đường quyết định (để viết "why" trong bài)

1. 16/09 — tra cứu §3j: 2/4 điểm "mới" đã có người làm (FAGER, AHEaD) → rút đóng góp còn (a) bộ prompt+contract Việt, (b) tầng sửa có quan sát ảnh phía sau Culture-TRIP.
2. 16/09 — A/B thật-vs-sinh kiểu FAGER: 0,58 vs mốc 0,97 → bộ chấm mù → sửa thang điểm trước khi đổi hành động sửa.
3. 17/09 — best-of-4 ≈ vòng lặp (khớp Ma et al. 2501.09732 và ablation T2I-Copilot) → bỏ vòng lặp, một lượt sửa; sau đó bỏ luôn gate (quyết định user 18/09).
4. 18/09 — img2img không đổi định danh vật → IP-Adapter (theo ImageRAG dùng đúng cấu hình SDXL + h94).
5. 19–20/09 — ablation refs-only/keep-only: refs gánh gain VCFS; agent giữ prompt/giảm confusable trên FLUX (README SAVIER "What the numbers say").

## 6. Giấy phép

Không LICENSE (chỉ đọc/trích dẫn, không chép mã): ImageRAG, Culture-TRIP, GenArtist, AHEaD, Gen-Searcher, ViFA-Council, ORIG.
MIT/Apache: SLD, FAGER, T2I-Copilot, RPG, MosAIG. Ảnh tham chiếu của nhóm: ảnh web bên thứ ba → không phát hành.
