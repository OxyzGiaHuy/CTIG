# Ảnh phẳng cho user study / figure

Cú pháp tên: `<prompt_id>_<model>_<arm>.png`
- `prompt_id`: S001–S050 (xem `data/prompts_simple.json` cho text VI/EN, entities)
- `model`: `sdxl` (SDXL 1.0) · `flux` (FLUX.1-dev)
- `arm`: `A` = prompt gốc · `I0` = Culture-TRIP refined prompt · `I1` = SAVIER (cùng seed với I0, + IP-Adapter 2 ảnh tham chiếu)

300 ảnh 1024×1024, trạng thái sau các sửa lỗi 21/09 (no-op có ref, xoá cụm theo mệnh đề, Evidence Card cứu JSON cụt trên SDXL).
Tạo bằng `scripts/export_results.py` rồi chép phẳng; refs-only/keep-only vẫn ở `results/<model>/<pid>/`.
