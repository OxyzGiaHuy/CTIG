# Phản biện nội bộ bản revision SOICT (agent research-critic, 2026-10-21)

Tóm tắt verdict theo từng đoạn sửa (đầy đủ trong transcript phiên làm việc):
- Giữ: contribution bullet 4 (+ reference-only ablation, + prompt-adherence); Table 2 (30 ô số đã kiểm khớp final_50.md); mục Prompt adherence (thêm câu CAIRE dùng cùng Qwen scorer); câu một seed/50 prompt.
- Sửa lại (đã áp): Abstract, Intro đoạn 4, Attribution, Runtime, Discussion, Conclusion viết lại đúng mức: CAIRE-VN +0.28 FLUX là 1/10 hiệu số có ý nghĩa (chưa hiệu chỉnh đa so sánh); CCR/VCFS trong nhiễu; SDXL kém về số ở mọi metric; giả thuyết encoder lẫn với cấu hình adapter; bỏ câu keep-only sai (keep-only cao hơn ở 3 ô); bỏ "below four plain samples" (tổng online 153 s > 146 s); bỏ C001 (n=1, chấm bằng mắt).
- Số sai đã sửa: CI I1−I0 và SAVIER−refs lấy từ README cũ → thay bằng final_50.md; "8→5 prompts below I0" là NRG<0 chứ không phải VCFS → thay bằng đếm VCFS thật (refs-only 7/8, SAVIER 8/10); 69.4/66.2 kiểm lại đúng (sensitivity bỏ 4 card rỗng, tính từ metrics_final2_*).
- Công bố thêm: 8 SDXL + 5 FLUX no-op có I1 ≡ I_ref; 4 lỗi Curator; CAIRE ablation cùng phiên; contract + photo bank do người duyệt.
- Mâu thuẫn còn sót đã sửa: "at most one" → "exactly one"; Intro bỏ "consistently"; Related work thêm câu định vị so với retrieval-based methods; caption Table 1 ghi chỉ hàng SAVIER có ảnh.
- Chưa làm (cần người): cắt về 12 trang (bỏ Fig 2 agents, Fig category, gộp Table 1+2, runtime vào văn bản); thay cột A bằng refs-only trong hình định tính (thầy từng yêu cầu bỏ cột này); trả lời R2 human verification bằng số contract/mục bị sửa so với bản LLM nháp (diff từ commit e1c9da0); README SAVIER còn câu "human pairwise preference is the primary evidence" và "6 categories" cần sửa.
- Đánh giá của critic: R1 attribution trả lời tốt về thiết kế, phần diễn giải trước đó tô hồng (nay đã hạ); R1 evaluator independence một phần; R1 seed và R2 human verification chỉ thừa nhận.
