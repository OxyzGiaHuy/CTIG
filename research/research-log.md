# Research log — CTIG

## 2026-09-11 — Bootstrap và smoke test v1
- Dựng pipeline v1 theo sơ đồ draft; chạy 2 prompt trên Kaggle 2×T4 (run `smoke`, 13,4 phút).
- Kết quả: 0/2 đạt sau 3 vòng. CLIP fidelity cuối 0,894, vòng 0 0,964 (giảm!).
- Soi log + ảnh: 7 lỗi, phần lớn là code:
  1. rút bằng chứng trả rỗng (bộ lọc khớp nguyên văn attr ↔ attr_sources)
  2. dịch trộn cấm vào bắt buộc ("wide obi at back" trong must_have của áo dài)
  3. prompt chuỗi lặp "Vietnamese Ao dai, Ao dai" mỗi vòng
  4. nhãn CLIP tiếng Việt ("Tết Trung Quốc") → CLIP vô nghĩa; ref Tết = pháo hoa Trung Quốc 0,996
  5. dịch thất bại để nguyên tiếng Việt vào prompt SDXL
  6. IP-Adapter 0,45 với ref CLIP 0,64 → nón lá neon khổng lồ
  7. Tết vắng cả 3 vòng không ai bắt; VLM bịa "bánh chưng đúng"
- Quyết định: sửa thành v1.1 trước khi chạy dev10. Giả thuyết H1–H7 ghi trong research-state.yaml.

## 2026-09-11 — v1.1
- Search: DuckDuckGo VI+EN không key; Commons thêm truy vấn VI; ref chỉ cho kind=object, CLIP ≥ 0,75.
- Rút bằng chứng: danh sách {attr, quote}, giữ khi quote có trong văn bản (khớp token ≥ 0,7).
- Tri giác: CLIP nhãn EN mô tả (KB thêm clip_label, name_en cho confusable); VLM checklist câu đóng.
- Phê bình: deterministic từ checklist; VLM không tự chấm điểm.
- Judge: BLIP-2 ITM + CLIP, độc lập với reviewer; lùi về CLIP nếu không tải được.
- Gen: prompt danh sách cụm, nhấn tối đa 1 lần; IP-Adapter 0,3; LCM-LoRA tuỳ chọn (fast_iters) + render đủ bước.
- Chưa chạy trên GPU. Bước tiếp: smoke v1.1 → dev10 với max_iters=0 (baseline) và max_iters=2 (H1).
