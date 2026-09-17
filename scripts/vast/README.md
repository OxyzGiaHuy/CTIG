# Script chạy trên máy thuê

Toàn bộ script từng chỉ tồn tại trên máy vast.ai, đưa vào git ngày 2026-09-17 khi chuyển máy. Đây là **biên
bản thí nghiệm**: mỗi tệp `run_*.sh` là đúng lệnh đã dùng cho một lượt chạy, mỗi tệp `*.py` là một phép đo.

Không tệp nào chứa khoá bí mật; chúng đọc `HF_TOKEN` từ `~/.bashrc` lúc chạy.

## Lệnh chạy theo phiên bản

| tệp | lượt chạy |
|---|---|
| `run_v171.sh` … `run_v194.sh` | các lượt v1.7 tới v1.9.4, vòng lặp cũ dựa trên KB |
| `run_label20.sh` | 20 prompt phủ 8 nhóm chủ đề, sinh ảnh để gán nhãn tay |
| `run_ctrip10.sh`, `run_ctrip_all.sh`, `run_ctrip14b.sh` | Culture-TRIP sinh prompt, bản 10 prompt và bản 100 prompt |
| `run_S001_3models.sh`, `run_S001_flux.sh`, `rerev_S001.sh` | S001 qua ba model nền, hai nhánh A và B |
| `run_loopv2_S012.sh` | Agentic Loop v2 trên S012 thuyền thúng |
| `queue_*.sh` | hàng đợi các lượt cũ, giữ để tra lại |

## Phép đo

| tệp | đo gì | kết luận đã rút ra |
|---|---|---|
| `probe_vqa.py`, `probe2.py` | so các cách hỏi VQA trên cùng một ảnh | câu có/không không tách được ảnh sai khỏi ảnh đúng; câu trắc nghiệm với mô tả sai viết tay thì tách được |
| `validate_fc.py` | câu trắc nghiệm với mô tả sai sinh tự động | AUC 0,29, dưới mức ngẫu nhiên; bỏ hướng này |
| `calib_width.py` | hiệu chỉnh ngưỡng nên dựa trên mấy ảnh thật | 3 ảnh `selected` tốt hơn 23 ảnh gồm cả `candidates` |
| `sweep_margin.py` | quét biên độ hiệu chỉnh | AUC 0,79 ở biên 0,08 |
| `crop_test.py`, `s012_crop.py` | cắt quanh chủ thể trước khi hỏi | AUC từng thuộc tính tăng rõ; cắt theo thực thể hơn cắt theo người |
| `score_rules.py`, `score_rules2.py` | ngưỡng cứng so với điểm mượt, tiêu chí loại thuộc tính | ngưỡng cứng cộng loại theo độ phân tán là tốt nhất |
| `replay_reviewer.py` | chạy lại Reviewer trên ảnh cũ | định lượng được AUC 0,00 của thang điểm cũ |
| `why.py`, `why2.py`, `verd.py` | đọc verdict của một ảnh cụ thể | dùng để soi vì sao một ảnh được điểm cao |
| `summ.py`, `summ_pairs.py`, `compare_pairs.py`, `rebuild_html.py` | tổng hợp kết quả, dựng trang so bare với system | |
