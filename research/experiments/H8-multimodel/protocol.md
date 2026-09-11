# H8 — So nhiều model sinh ảnh trên cùng GenSpec

**Câu hỏi:** Với cùng prompt/negative đã có bằng chứng văn hoá, model nào vẽ thực thể Việt đúng hơn, và LoRA áo dài
có nâng CLIP identity của áo dài so với SDXL gốc không?

**Thay đổi:** `models: [sdxl_turbo, dreamshaper8, sdxl_base, sdxl_aodai, playground25]`, cùng GenSpec, seed 1234, 2 ứng viên.
**Giữ cố định:** config `kaggle_walkthrough_t4x2.yaml`, tập dev10, review tắt.
**Metric chính:** CLIP identity của ứng viên được chọn, theo model, trung bình trên dev10 (chỉ thực thể object).
**Metric phụ:** BLIP-2 ITM, thời gian/ảnh, VRAM đỉnh; đối chiếu bằng mắt trong bundle.html / grid.png.
**Dự đoán (khoá trước khi chạy):**
1. `sdxl_aodai` > `sdxl_base` về CLIP identity trên các prompt có áo dài (p001, p036, p042, p049, p050).
2. `sdxl_turbo` thấp hơn `sdxl_base` (không negative, 4 bước) trên thực thể có confusable mạnh (bánh chưng, đàn bầu).
3. `dreamshaper8` (SD1.5) thấp nhất về danh tính văn hoá dù ảnh "đẹp".
**Điều gì bác bỏ:** LoRA không tăng hoặc giảm CLIP identity; hoặc tăng nhưng mắt người thấy áo dài xấu/khác hẳn (metric sai).
**Loại:** CONFIRMATORY cho 1, EXPLORATORY cho 2-3.
