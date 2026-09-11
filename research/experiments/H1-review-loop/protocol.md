# H1 — Vòng review nâng chất lượng văn hoá so với sinh trực tiếp

**Thay đổi:** `review.max_iters` 0 (baseline, chỉ vòng 0) so với 2.
**Giữ cố định:** config `kaggle_t4x2.yaml`, seed 1234, dev10, cùng ngày chạy.
**Metric chính:** mean CLIP fidelity của ảnh cuối trên dev10 (chỉ thực thể kind=object).
**Metric phụ:** tỉ lệ đạt theo checklist, số vòng trung bình, thời gian.
**Dự đoán (khoá trước khi chạy):** CLIP fidelity cuối tăng ≥ 0,10 so với baseline; tỉ lệ đạt tăng.
**Điều gì bác bỏ:** CLIP fidelity không tăng hoặc giảm; hoặc tăng nhưng ảnh nhìn tệ hơn trong bundle.html
(khi đó metric sai, không phải giả thuyết đúng).
**Loại kết quả:** CONFIRMATORY.
