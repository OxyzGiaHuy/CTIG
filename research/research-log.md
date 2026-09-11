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

## 2026-09-11 — Smoke v1.1 và quyết định đổi trọng tâm sang v1.2
- Smoke v1.1 (2 prompt, 2×T4, 12,4 phút): 0/2 đạt. CLIP fidelity vòng 0 → cuối 0,60 → 0,74; ảnh p050 vòng 2 đúng văn hoá
  (áo dài + nón lá) nhưng bộ chấm không nhận ra vì spec hỏng.
- 5 lỗi mới: (1) analysis trả 37/37 thực thể KB cho "mừng năm mới âm lịch" → spec giữ áo bà ba, cắt Tết; (2) negative chứa
  "áo dài, vietnamese, tunic" do confusable nội bộ Việt và băm token; (3) dịch EN trả rỗng không log; (4) VLM trả "yes" cho
  mọi câu cấm trên áo dài đúng → 4 critical → điểm 0; (5) ~40% thuộc tính rút được là tên loại ("Hình dạng", "Chất liệu").
- Bằng chứng tốt: rút bằng chứng chạy được (36 thực thể, có câu gốc); BLIP-2 ITM nạp được và chấm 0,94 khớp mắt người ở p001
  trong khi reviewer chấm 0.
- **Quyết định:** v1.2 đổi trọng tâm sang notebook hiển thị từng bước + so nhiều model; review agent thành cờ, mặc định tắt.

## 2026-09-11 — v1.2
- `Session` memo từng bước (bộ nhớ → đĩa → chạy mới) + cache mọi lần gọi LLM (`runs/_cache/llm`) + cache web (`runs/_cache/web`):
  chạy lại cell không tốn API/model.
- Search so sánh hai cột: truy vấn từ keywords vs từ prompt gốc, top-K text/ảnh với CLIP sim.
- Multigen: registry (sdxl_turbo, dreamshaper8, sdxl_base, sdxl_aodai [Civitai LoRA 590793], playground25; sd3/hunyuan
  experimental), nạp tuần tự, ảnh dùng lại theo hash GenSpec, grid + CLIP identity + BLIP-2 ITM + CLIP sim.
- Sửa 5 lỗi v1.1: cap 6 ứng viên có căn cứ (context ưu tiên), negative chỉ confusable khác văn hoá và tên ASCII,
  KB thêm must_have_en/must_not_en viết tay (KB 0.3.0), lọc rác rút bằng chứng, forbidden-yes khi identity=target hạ mức.
- Chưa chạy GPU. Kế tiếp: walkthrough p001/p050/p012 trên Kaggle, rồi H8 trên dev10.
