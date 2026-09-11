# Findings — CTIG (tổng hợp vòng ngoài, cập nhật sau mỗi 5–10 run)

## Đã có bằng chứng
- **Vòng review bắt được lỗi vắng mặt.** p050 vòng 0 không có áo dài, VLM báo đúng, vòng 1 có áo dài. (smoke v1, n=2, chưa đủ để kết luận)
- **CLIP với nhãn trần không đáng tin cho danh tính khi nhãn không phải tiếng Anh mô tả.** "Tết Trung Quốc" 0,01 vs "Vietnamese Lunar New Year" 0,99 trên ảnh không có gì liên quan Tết.
- **Qwen2.5-VL-3B viết findings tự do không dùng được**: điểm cố định 0,6, lý giải lặp, judge trả lời bằng tiếng Trung. Trả lời câu hỏi đóng chưa đo.

## Chưa có bằng chứng (đang là giả thuyết)
- Vòng review nâng CLIP fidelity trên tập dev (H1). Smoke v1 cho thấy vòng 0 cao hơn vòng cuối, nhưng do bug prompt.
- Web tiếng Việt cho must_have thị giác nhiều hơn Wikipedia (H2). Đã thấy ví dụ định tính với "nón lá", chưa đo.
- BLIP-2 ITM làm judge tốt hơn Qwen (H3). Chưa có user study để so.

## Kết quả âm đáng giữ
- Từ bản mô phỏng: reviewer thứ hai (VisualCritic) gần như không đổi kết quả (0,857 → 0,837). Chưa kiểm trên model thật.

## Câu hỏi mở cho vòng ngoài kế tiếp
- Với thực thể prior thấp, vòng review có hội tụ không hay chỉ tốn 3 lần sinh? (liên quan H5, quyết định có làm LoRA hay không)
