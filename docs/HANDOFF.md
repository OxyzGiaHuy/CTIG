# HANDOFF — CTIG, trạng thái 2026-09-16

Tài liệu bàn giao cho người tiếp tục (hoặc cho chính nhóm sau khi mở lại máy). Đọc cùng `docs/FLOW_v1.7.md` (flow và model
từng bước), `research/research-log.md` (nhật ký theo ngày), `research/research-state.yaml` (giả thuyết H1–H21).

## 1. Phiên bản code và ý nghĩa

| tag | commit | nội dung | trạng thái chạy |
|---|---|---|---|
| v1.7 | `5b770e6` | Grounding gom bước; hàng `M#bare` cùng seed; Agentic Review Loop (Reviewer / Reflector / Refiner, 3 vòng, giữ hết ảnh, chọn trên pool) | đã chạy 4 prompt gốc + 4 complex trên A100 |
| v1.7.1 | `862df1c`… | Reviewer: VQA có/không từng thuộc tính, VQAScore cột tham chiếu, trọng số định danh, "đạt" = đủ mọi must_have; render `caption`; Analysis alias dài che alias ngắn; Filter bỏ thực thể w<0,6; spec bỏ must_not nhắc lại must_have | đã chạy, số liệu trong research-log |
| v1.7.2 | `59c56af`, `5192539` | một model nền = một hệ thống: Rank / loop / ảnh cuối chạy riêng theo checkpoint (`per_model`), nhãn nhóm = khoá model nền không LoRA/IP-Adapter | chạy 6/8 prompt rồi dừng chủ động (p050, C037 chưa) |
| v1.8 (chưa chạy) | `56aa202`, `da0106b`, `5bf4f1a`, `3130d29` | họ FLUX.1-dev và SD 3.5 Medium; nhóm M mới; KB tự sinh trong Grounding (`kb_mode: auto`); bộ prompt S/C của nhóm | **chưa smoke test** (bị cắt khi tắt máy) |

Run trên máy vast: `runs/v17` (4 prompt gốc), `runs/v17_complex` (C002, C003, C008, C037 bản cũ), `runs/v18_smoke` (S001 mới bắt đầu).
Các lượt v1.7.1 và v1.7.2 **ghi đè** `runs/v17`; bản đã copy về máy local: `~/Research/VnCultureGen/result_kaggle/vast_v1.7/`
(`bare_vs_system_v171.html`, `bare_vs_system_final.html`, `bare_vs_system_complex_final.html`, `progress_report_final.html`).
Từ v1.8 mỗi phiên bản dùng run-name riêng (`v18`, `v18_complex`).

## 2. Kết quả chính đã có (v1.7.1, 4 model nền cũ)

- Prompt gốc (p001 áo dài, p012 thuyền thúng, p031 áo tứ thân, p050 Tết): system > bare ở 15/16 cặp; Reviewer TB bare→system:
  SD1.5 0,17→0,17, DreamShaper 0,13→0,33, SDXL 0,11→0,37, RealVis 0,08→0,30. ITM attr tăng 3/4 (SD1.5 không).
- Complex (C002, C003, C008, C037 bản cũ): 12/16 cặp; chỉ DreamShaper và SDXL tăng rõ; RealVis bằng; SD1.5 kém nhẹ.
- Kết luận tạm: hệ thống bổ trợ nhưng không vượt năng lực model nền; lợi thế co lại ngoài KB tay (thiên lệch KB) → lý do làm KB tự sinh và
  thêm SD 3.5 / FLUX.
- Loop cải thiện thật ở p050 (vòng 2) và C002 (vòng 2); 6/8 prompt dừng sau 2 vòng không tăng.
- PickScore chấm ảnh bare đẹp hơn (0,83 so với 0,28) và VQAScore bão hoà ~0,89 kể cả ảnh sai → chỉ làm cột tham chiếu.
- p031 áo tứ thân: không model nào vẽ được bốn tà; ca cho LoRA tự huấn luyện.

## 3. Lỗi đã gặp và đã sửa (để không lặp lại)

| lỗi | biểu hiện | sửa |
|---|---|---|
| top-k Reviewer theo ensemble | PickScore đẩy 5/8 ảnh bare vào top-k, hàng system không được chấm | Reviewer hai tầng: VLM chấm mọi ảnh, metric chỉ xếp trong tập đã qua |
| mốc cải thiện = top-1 Rank | vòng 1 "tốt hơn" giả (p012) | mốc = điểm Reviewer cao nhất trong pool |
| Reflector không leo nấc | 3 vòng cùng `attr_refs` | xét lần gần nhất của cách sửa; thang: ground_refs → attr_refs → more_refs → seed → guidance |
| `adapt_spec` reset `iteration=0` | mọi vòng sinh lại cùng seed (ẩn từ v1.4) | giữ iteration |
| ảnh cuối rơi vào hàng bare | p031, p050 | bare chỉ để so, không vào Rank/loop/pool |
| pool gom mọi model | ensemble giữa model, trái mục tiêu 1 model | loop riêng theo checkpoint (`per_model`) |
| web thêm must_not "conical shape" cho nón lá | VQA loại mọi ảnh nón lá đúng | spec bỏ must_not chỉ nhắc lại must_have; KB áo tứ thân bỏ "đi với nón quai thao" |
| "Tết Trung Thu" kéo theo Tết Nguyên Đán | C037 đòi quất, lì xì | alias dài che alias ngắn |
| áo dài w=0,55 lọt Filter ở C008 | đòi cổ đứng trong prompt áo tứ thân | Filter chỉ thực thể w≥0,6 |
| Reviewer loại hết → ảnh cuối None | p031 | lấy ảnh ít sai nhất làm mốc cho loop |
| must_not "wide brim" theo VQA 0,78 trên nón lá đúng | loại nhầm | must_not chỉ theo VQA cần ≥0,85 |
| `pkill -f`/`pgrep -f` giết chính phiên ssh | mất output 3 lần | dùng `[r]un_` hoặc ghép chuỗi pattern |
| nhãn nhóm = hàng có điểm cao nhất (`sdxl_refplus`) | cột "ảnh cuối loop" không khớp bảng bare/system | nhãn = khoá model nền không LoRA/IP-Adapter |
| chèn code bằng `replace(anchor, new+anchor)` | dòng `def` nhân đôi → SyntaxError (2 lần) | kiểm `ast.parse` mọi module trước commit |

## 4. Lỗi/rủi ro còn mở

- **KB tự sinh chưa chạy thật**: chất lượng thuộc tính Qwen 3B viết cho thực thể hiếm, tỉ lệ bản nháp bị loại vì thiếu câu gốc.
  Review code (2026-09-16) đã sửa: lọc rác `clean_extracted`, cache tôn trọng `evidence_cache`, không đóng băng bản mỏng, `rehydrate`
  cho Session và Pipeline, `kind` enum, backend API nhận `max_new_tokens`. Còn để ý: chi phí ~60–90 s/thực thể lần đầu (38 thực thể tay
  cũng dựng lại trong `kb_mode: auto`), prompt 6×2500 chữ có thể rút xuống 4×1800.
- **FLUX / SD 3.5 chưa chạy**: IP-Adapter XLabs cho FLUX chưa thử (`negative_ip_adapter_image`, `true_cfg_scale`), hires tắt cho họ DiT,
  `prepare_ip_adapter_image_embeds` khác chữ ký (đã có fallback). Đĩa vast còn ~33 GB sau khi tải FLUX.
- **Prompt caption**: thua legacy ở p001, thắng ở C002/C003 → chưa chốt; hàng `realvis_xl#caption` vẫn trong config.
- **Reviewer ở prompt complex**: ITM attr ≈ 0, CLIP attr ≈ 1,0 → chỉ điểm Reviewer dùng được; cần đánh giá người theo cặp.
- **Sinh và chấm cùng nguồn thuộc tính** (KB hoặc bản tự sinh): hệ thống được cộng điểm vì vẽ đúng checklist của mình. Hướng: Reviewer
  chấm theo brief Wikipedia độc lập với prompt; tập kiểm ngoài KB; đánh giá người.
- **`best_model` lệch ảnh cuối** trong một vài hồ sơ cũ (nhãn model của top-1 Rank thay vì của ảnh cuối) — đã sửa ở v1.7.2.
- **Bộ complex đổi câu** ở C002, C003, C008 (nhóm sửa 2026-09-15) → số v1.7 complex không so trực tiếp được với v1.8.

## 5. Máy vast.ai

1× A100 80 GB (Taiwan), đĩa 100 GB. Repo `/workspace/ctig17` (git pull), venv `/venv/main`, `HF_HOME=/workspace/.hf_home`,
`HF_TOKEN` trong `~/.bashrc` (bashrc return sớm ở shell không tương tác → `export HF_TOKEN=$(grep "^export HF_TOKEN" ~/.bashrc | cut -d= -f2)`).
Kho ảnh `/workspace/refs` (1.399), index `/workspace/runs/_cache/ref_index.npz`. Log `/workspace/logs/`. Script phụ `/workspace/{summ.py,
summ_pairs.py,verd.py,compare_pairs.py}`. Chạy nền bằng `setsid nohup ... &`. Host/port SSH đổi sau mỗi Start; Stop giữ đĩa, Destroy mất.

## 6. Việc kế tiếp theo thứ tự

1. Mở máy → `git pull` → smoke SD 3.5 Medium (S001, S012; 5 hàng) → smoke FLUX (3 hàng) → sửa lỗi họ DiT.
2. Chạy v1.8: S001, S012, S031, S023, S021 + C002, C003, C008, C037 (bản mới), hai luồng song song, run `v18` / `v18_complex`,
   `kb_mode: auto` → xem `runs/_cache/kb_auto/` và `scripts/kb_auto_export.py` cho nhóm duyệt.
3. `scripts/compare_pairs.py` cho hai run; cập nhật research-log, research-state (H20, H21, thêm H22 "negative có cần không" từ FLUX).
4. Sau đó: dev set rộng hơn (3 luồng), LoRA áo tứ thân, đánh giá người theo cặp, tách nguồn chấm khỏi nguồn sinh.
