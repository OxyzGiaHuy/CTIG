# Danh sách đọc theo từng khối của CTIG (cập nhật 2026-09-13)

Mỗi mục: bài báo, một câu họ làm gì, và **CTIG lấy gì**. Đọc chung: bắt đầu từ các mục đánh ★.

## 0. Toàn hệ thống: pipeline văn hoá / agentic T2I

| | bài | họ làm gì | CTIG lấy gì |
|---|---|---|---|
| ★ | [Culture-TRIP (NAACL 2025)](https://arxiv.org/abs/2502.16902) | Truy hồi Wikipedia/web về "culture noun" trong prompt, LLM sửa prompt lặp theo bộ tiêu chí văn hoá, khảo sát 66 người / 8 nước. Vòng lặp ở văn bản, không nhìn ảnh. | Kiến trúc Search → Spec → prompt; Summary agent (2c) và cờ `enrich_prompt`. Khác: CTIG có kênh ảnh và chấm điểm mức thuộc tính. |
| ★ | [ImageRAG (ICLR 2026)](https://arxiv.org/abs/2502.09411) | VLM hỏi "ảnh có khớp prompt không"; không thì liệt kê khái niệm thiếu, viết caption truy hồi, lấy ảnh từ LAION 350k bằng CLIP, đưa vào SDXL qua IP-Adapter scale 0.5 (≤ 3 ảnh, 1 ảnh/khái niệm). 84–92 % thành công trên khái niệm hiếm. Hạn chế: truy hồi sai, CLIP đếm kém. | **Retrieval có điều kiện**: chỉ dùng ảnh khi model không tự vẽ được (v1.5: gate theo prior + theo Filter). Xác nhận IP-Adapter là kênh đúng, scale 0.4–0.5. |
| | [MosAIG / When Cultures Meet (2025)](https://arxiv.org/abs/2502.15972) | Đa tác tử (Moderator, 3 Social Agents, Summarizer) viết caption giàu văn hoá cho prompt đa văn hoá; benchmark 9k ảnh 5 nước có Việt Nam; prompt tiếng Việt cho alignment thấp nhất. | Bằng chứng "tiếng Việt vào thẳng model là tệ nhất" → dịch + làm giàu là bắt buộc. Cách tách vai agent theo khía cạnh. |
| | [T2I-Copilot (ICCV 2025)](https://arxiv.org/abs/2507.20536) | 3 agent: Input Interpreter, Generation Engine (chọn model), Quality Evaluator; training-free; VQAScore ngang model thương mại. | Chọn model theo prompt là một agent; CTIG hiện chạy mọi model rồi chọn (best-of-models). |
| | [RPG (ICML 2024)](https://arxiv.org/abs/2401.11708) | MLLM tách prompt phức thành vùng, recaption từng vùng, khuếch tán theo vùng. | Hướng cho bộ prompt phức hợp (nhiều thực thể một cảnh): sinh theo vùng thay vì một prompt. |
| | [Marmot (2025)](https://arxiv.org/abs/2504.20054) | Decision-maker / Executor / Verifier với model 8B; hai tầng: MLLM mô tả từng vật thể, LLM so văn bản; giới hạn vòng. | Filter agent hai tầng (mô tả rồi so văn bản), giới hạn 1 vòng sửa. |
| | [GenArtist (NeurIPS 2024)](https://arxiv.org/abs/2407.05600), [SLD (2023)](https://arxiv.org/abs/2311.16090), [Idea2Img (ECCV 2024)](https://link.springer.com/chapter/10.1007/978-3-031-72920-1_10) | MLLM làm agent điều phối công cụ sinh/sửa, tự kiểm từng bước; SLD dùng detector + LLM sửa vị trí vật thể; Idea2Img lặp với GPT-4V. | Mẫu "verify từng bước có công cụ"; SLD gợi ý detector cho vị trí/số lượng. |

## 1. Khối Analysis (prompt → thực thể văn hoá)

| | bài | CTIG lấy gì |
|---|---|---|
| | Culture-TRIP (mục 0) — "culture noun" là đơn vị truy hồi | Thực thể văn hoá là đơn vị của mọi bước sau; giới hạn 6 ứng viên có căn cứ. |
| | RPG (mục 0) — recaption từng thực thể | Cho bộ prompt phức hợp: mỗi thực thể một mô tả riêng trước khi ghép. |

## 2. Khối Search / Retrieval-augmented generation

| | bài | họ làm gì | CTIG lấy gì |
|---|---|---|---|
| | [Re-Imagen (2022)](https://arxiv.org/abs/2209.14491) | Huấn luyện denoiser nhận cặp text-ảnh truy hồi; tốt cho thực thể hiếm. | Tên gọi đúng của bài toán: retrieval-augmented T2I. CTIG không huấn luyện lại mà dùng IP-Adapter. |
| | [RDM (Blattmann 2022)](https://arxiv.org/abs/2204.11824), kNN-Diffusion | Điều kiện hoá bằng CLIP embedding của ảnh láng giềng. | IP-Adapter là bản "không huấn luyện" của ý này. |
| | [RealRAG (2025)](https://arxiv.org/abs/2502.00848) | Truy hồi ảnh thật để sinh vật thể hiếm/mới, học tương phản tự phản tư. | Ảnh thật làm mốc cho vật thể prior thấp (thuyền thúng). |
| ★ | ImageRAG (mục 0) | Truy hồi **có điều kiện**, theo khái niệm thiếu. | v1.5: `auto_ref` — chỉ cấp ảnh tham chiếu cho thực thể prior thấp hoặc khi Filter báo thiếu thuộc tính. |

## 3. Khối Bằng chứng / Spec / Prompt

| | bài | CTIG lấy gì |
|---|---|---|
| ★ | [CULTIVate (2025)](https://arxiv.org/abs/2511.05681) | Descriptor tham chiếu theo 5 chiều (background, attire, objects, interactions, layout) do LLM đề xuất rồi lọc; 90 % chính xác. | v1.5: brief của Summary agent có 4 chiều này. must_have/must_not của CTIG là descriptor viết tay. |
| | Culture-TRIP (mục 0) — tiêu chí văn hoá để sửa prompt | Vòng sửa 4d. |
| | [DALL·E 3 prompt upsampling], Promptist | Viết lại prompt thành caption chi tiết; RL theo aesthetic/CLIP. | Render `sentence` cho SD3/FLUX; không dùng RL vì reward chưa đo "đúng". |

## 4. Khối Sinh ảnh (tiêm tri thức)

| | bài | họ làm gì | CTIG lấy gì |
|---|---|---|---|
| ★ | [SCoFT + CCUB (CVPR 2024)](https://arxiv.org/abs/2401.08053) | Bộ dữ liệu văn hoá nhỏ do cộng đồng chọn (CCUB), fine-tune SD bằng self-contrastive loss dùng chính thiên lệch của model làm âm; khảo sát 51 người / 5 nước. | Con đường LoRA/fine-tune từ ảnh đã lọc (H5): loss tương phản với ảnh "sai kiểu qipao" mà model tự sinh. |
| | [IP-Adapter (2023)](https://arxiv.org/abs/2308.06721) | Cross-attention tách cho ảnh; Plus dùng ViT-H, nhiều ảnh. | Kênh ảnh hiện tại; scale 0.4; cắt ảnh theo thực thể để tránh chép bố cục. |
| | [DreamBooth (2022)](https://arxiv.org/abs/2208.12242) + LoRA | Vài ảnh → token mới cho concept. | LoRA áo dài Civitai đang dùng; tự train cho thực thể không có LoRA. |
| | [CultureCLIP (2025)](https://arxiv.org/abs/2507.06210) | Fine-tune CLIP trên cặp "trông giống nhau nhưng khác văn hoá" (CulTwin) → +5,5 % nhận diện tinh. | CLIP-B/32 của CTIG bão hoà danh tính; CultureCLIP (nếu có trọng số) là ứng viên thay cho CLIP id/attr. |

## 5. Khối Chấm điểm / Đánh giá

| | bài | họ làm gì | CTIG lấy gì |
|---|---|---|---|
| ★ | [DSG (ICLR 2024)](https://arxiv.org/abs/2310.18235), [TIFA](https://arxiv.org/abs/2303.11897), Soft-TIFA, [VQAScore](https://arxiv.org/abs/2404.01291) | Tách prompt thành câu hỏi nguyên tử, VQA trả lời; DSG thêm DAG phụ thuộc (không hỏi "có đỏ không" khi vật không có). | Checklist thuộc tính của CTIG là TIFA-kiểu; ITM attr = Soft-TIFA với BLIP-2. DAG: chỉ chấm thuộc tính khi danh tính có (v1.5 ghi chú, chưa làm). |
| ★ | [Ma et al., Inference-time scaling (CVPR 2025)](https://arxiv.org/abs/2501.09732) | Best-of-N và tìm nhiễu theo verifier; verifier đơn bị "hack" (Aesthetic thiên style, CLIP thiên chữ) → **Verifier Ensemble** = trung bình hạng không trọng số. | v1.5: điểm tổng = trung bình hạng (CLIP id, attr, ITM attr, PickScore) thay trung bình giá trị; best-of-N thích nghi (thêm ứng viên khi verifier chưa đạt). |
| ★ | [CulturalFrames (2025)](https://arxiv.org/abs/2506.08835) | 983 prompt, 10 nước, kỳ vọng văn hoá tường minh/ngầm; model sai 44 %; metric tự động tương quan yếu (tốt nhất ~0,31, người-người 0,38); CLIPScore tệ nhất. | Cảnh báo: mọi metric của CTIG cần user study đối chiếu; "kỳ vọng ngầm" (Tết phải có bánh chưng dù prompt không nói) đúng là must_have của thực thể context. |
| | CULTIVate (mục 3) — Alignment / Hallucination / Exaggeration | ref_sim và must_not đóng vai Hallucination; chưa có Exaggeration (phóng đại stereotype). |
| | [CuRe (2025)](https://arxiv.org/abs/2506.08071) | Marginal utility khi thêm thuộc tính vào prompt làm proxy cho người. | Cách đo "thêm must_have có đổi ảnh không" cho bộ prompt cuối. |
| | [CultDiff / Global Lens (2025)](https://arxiv.org/abs/2502.08914) | Metric ảnh-ảnh học từ phản hồi người; 10 nước. | Ảnh Commons đã lọc làm mốc ảnh-ảnh, cùng ref_sim. |
| | [CUBE (2024)](https://openreview.net/forum?id=4351SumKS9), GlobalRG, [Community-informed rubrics (2026)](https://arxiv.org/abs/2604.02406) | Benchmark cultural competence 8 nước; rubric do cộng đồng định nghĩa trước khi tự động hoá. | Cách xây rubric user study cho bộ prompt cuối: cộng đồng định nghĩa "đúng" trước, VLM-judge sau. |
| | [PickScore (2023)](https://arxiv.org/abs/2305.01569), ImageReward, HPSv2 | Reward theo sở thích người. | Cột "đẹp"; chỉ là một verifier trong ensemble. |

## 6. Công cụ phụ trợ

| | bài / công cụ | CTIG lấy gì |
|---|---|---|
| | [OWL-ViT (ECCV 2022)](https://arxiv.org/abs/2205.06230), Grounding DINO | Phát hiện vật thể theo câu chữ → v1.5 cắt ảnh tham chiếu bằng OWL-ViT, lùi về CLIP quét lưới. |
| | [VietFashion (2026)](https://arxiv.org/html/2606.13427) | Benchmark truy hồi áo dài với từ vựng thuộc tính từ tạp chí thời trang → nguồn thuộc tính thị giác tiếng Việt cho KB. |

## Kết luận rút ra cho v1.5

1. Truy hồi ảnh phải **có điều kiện** (ImageRAG): prior thấp hoặc Filter báo thiếu thì mới dùng ảnh; prior cao thì ảnh chỉ mang rủi ro chép.
2. Verifier đơn bị hack (Ma et al.): xếp hạng bằng **trung bình hạng** nhiều verifier, thêm ứng viên khi verifier chưa đạt.
3. Metric tự động tương quan yếu với người (CulturalFrames): user study trên bộ prompt cuối là bắt buộc, không phải phụ.
4. Descriptor theo chiều (CULTIVate): brief và must_have nên gắn chiều attire / objects / background / interactions.
5. Fine-tune từ ảnh cộng đồng chọn (SCoFT): con đường LoRA cho thực thể không ai làm LoRA sẵn.
