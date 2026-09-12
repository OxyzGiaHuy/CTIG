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

## 2026-09-11 — Walkthrough v1.2 lần đầu trên Kaggle (p001, 2×T4)
- Chạy trọn vẹn: 4/5 model ra ảnh (turbo 48s/7,1 GB, dreamshaper 20s/2,6 GB, sdxl 86s/7,1 GB, playground 100s/9,6 GB).
  Cache: 3 lần gọi Qwen, 14 truy vấn web. Hàng `sdxl_aodai` lỗi ở bước GẮN LoRA dù file tải đúng (80 MB, 2166 tensor kohya):
  nghi thiếu gói `peft`; code nuốt thông điệp gốc.
- Search hai cột: truy vấn từ keywords ra Wikipedia + bài "Cấu tạo của áo dài"; truy vấn từ prompt gốc ra tin lá cải và
  Shutterstock. Ngược lại với ẢNH: cột prompt gốc ra ảnh nữ sinh áo dài trắng khớp hơn (sim 0,37 vs 0,32).
- Grid: SDXL c0 và DreamShaper c0 là áo dài đúng có quần; Turbo c1 có đai đỏ (nghiêng qipao); Playground ra váy liền.
  Nhưng CLIP identity 0,95–1,00 và ITM 0,92–1,00 cho tất cả → **thước đo danh tính bão hoà** trên prompt dễ.
- Rút bằng chứng: 3 must_have thì 2 trùng y chữ; DDG chỉ trả snippet 100–340 ký tự nên model chỉ có một câu Wikipedia để rút; 73 s.
- Dịch: Qwen trả dict `{"ao_dai": {...}}` thay vì `{"entities": [...]}` → "0 thực thể".
- **v1.2.1:** LoRA có fallback fuse + giữ lỗi gốc + `peft` vào requirements; nhận cả hai dạng JSON dịch; tải toàn văn top-3 trang
  web thay snippet, bỏ trùng thuộc tính, cap 6 nguồn; thêm điểm mức thuộc tính (CLIP tương phản must_have/must_not, ITM
  thuộc tính) và điểm tổng; thêm hàng `sdxl_ref` (SDXL + IP-Adapter ảnh Commons) để kiểm H4 trên cùng grid.

## 2026-09-11 — Walkthrough v1.2.1 trên Kaggle (p001, 2×T4)
- Hai lỗi hạ tầng trước khi chạy được: (1) Kaggle tự giải nén `runs_cache.zip` khi upload Dataset → cell khôi phục
  phải copy thư mục, không `zipfile`; (2) `torchao 0.10` có sẵn trên ảnh Kaggle làm `peft` import chết → LoRA lỗi
  lần hai với lý do khác lần một. Gỡ torchao là gắn được. Bài học: cache bước (`step_multigen.json`) đã đóng băng hàng
  lỗi nên sửa xong vẫn thấy lỗi → Session nay từ chối dùng lại bước có hàng lỗi.
- **6/6 model ra ảnh.** LoRA áo dài gắn qua peft (84 s, 7,15 GB). `sdxl_ref` IP-Adapter 0,3 với ảnh Commons đạt CLIP 0,99
  (94 s, đỉnh 11,2 GB — sát T4). Playground 91 s, 7,1 GB sau khi dùng VAE fp16-fix.
- **Thước đo mức thuộc tính tách được ảnh mà danh tính không tách được.** CLIP identity 0,92–1,00 và ITM 0,92–1,00 cho cả
  12 ảnh. CLIP attr 0,14–0,67: các ảnh váy xẻ tà KHÔNG QUẦN (dreamshaper c1 0,18; sdxl_base c1 0,40; sdxl_ref c1 0,33;
  playground cả hai 0,14/0,18) đều thấp hơn ảnh có quần cùng model (0,67; 0,45; 0,51). Đối chiếu bằng mắt khớp.
- **Lỗi văn hoá thực tế của p001 không phải kimono mà là "qipao hoá": váy liền xẻ tà cao, không quần** (4/12 ảnh) và
  đai đỏ (turbo c1). Negative hiện chỉ có tên confusable (kimono, qipao, hanbok) không chặn được. → H9: đưa must_not_en
  ("one-piece dress with no trousers underneath") vào negative; đồng thời prompt chỉ lấy 2 must_have nên "worn over
  wide-legged long trousers" (thứ 3) bị cắt → mặc định 3.
- **Phương sai theo seed lớn hơn phương sai giữa model.** Cùng sdxl_ref: attr 0,51 vs 0,33; ITM attr 0,87 vs 0,33.
  Với n=2 không kết luận được model nào tốt hơn; so model cần n ≥ 4 và nhiều prompt (H8 sửa prediction).
- Rút bằng chứng từ toàn văn 3 trang: 2 must_have rút được đều chỉ nói lại KB ("xẻ hai tà") rồi được Qwen dịch tệ
  ("cut into two parts") và lọt vào spec. → thuộc tính trùng KB được đếm là "xác nhận KB", không thành thuộc tính mới.
- Nhỏ: grid.png ra ô vuông thay dấu tiếng Việt (Kaggle không có DejaVu hệ thống) → dùng font của matplotlib;
  `cổng trường` (tiếng Việt) lọt vào prompt SDXL qua scene_notes → chỉ giữ ghi chú ASCII.
- **v1.2.1 (code):** must_not_en vào negative, 3 thuộc tính/thực thể, lọc xác nhận KB, font grid, khoá cache bước có
  số phiên bản logic (đổi code → bước chạy lại, không cần xoá cache tay).
- Kế tiếp: chạy lại p001 (so cùng seed: có/không negative must_not → H9), rồi p050, p012, p031.

## 2026-09-12 — v1.3: tối ưu ảnh cuối trên T4 (chưa chạy GPU)
- Người dùng hỏi "tối ưu nhất có thể trên Kaggle để ảnh cuối đẹp và chuẩn, chưa xét agentic". Quyết định theo thứ tự hiệu quả
  trên chi phí: (1) best-of-N + bộ chấm thẩm mỹ, (2) scheduler/guidance/compel, (3) checkpoint tốt hơn, (4) hires fix,
  (5) IP-Adapter Plus nhiều ảnh + sweep LoRA scale. Mỗi thứ là một cờ hoặc một hàng riêng để so cùng seed (H10–H12).
- Thêm PickScore v1 (CLIP-H tinh chỉnh theo sở thích người) làm cột "đẹp", chuẩn hoá min-max trong lần chạy; nó đo "thích",
  không đo đúng văn hoá, nên đứng cạnh CLIP attr chứ không thay.
- Phát hiện lỗi cũ khi viết test: `multigen.json` bị ghi đè sau mỗi hàng rồi đọc lại cho hàng kế → chỉ hàng đầu tiên được dùng
  lại ảnh trên đĩa; các hàng sau sinh lại dù GenSpec không đổi (log v1.2.1 xác nhận: chỉ turbo "ảnh từ lần trước"). Sửa: đọc
  một lần ở đầu run(). Đây là lý do lần chạy v1.2.1 tốn 6 phút thay vì ~3.
- Rủi ro chưa kiểm trên GPU và đường lùi tương ứng: compel + cpu offload (embeds bị từ chối → prompt thô); nhiều ảnh cho một
  IP-Adapter (bị từ chối → một ảnh); img2img 1536px OOM (→ ảnh gốc, tắt hires cho hàng đó); SD3.5 fp16 trên T4 (experimental).
- Kế tiếp: chạy p001 với config t4x2 v1.3; so grid cùng seed với v1.2.1 (H9 negative must_not, H11 hires, H12 RealVisXL);
  rồi p031, p050, p012.

## 2026-09-12 — Walkthrough v1.3 trên Kaggle (p001, 2×T4): 7 model × 4 ứng viên, hires, ~25 phút
- **H9 (negative must_not) có tác dụng rõ:** 28 ảnh thì gần như tất cả có quần dài (v1.2.1: 4/12 không quần). Tác dụng phụ:
  dreamshaper8 c3/c4 thành bộ vest trắng cổ đứng + quần (quá tay), sdxl_base ra "áo khoác dài + quần" thay áo dài ôm. CLIP attr
  vẫn xếp đúng các ca này xuống dưới (0,42; 0,46).
- **H12 (RealVisXL > SDXL base) được ủng hộ trên p001:** attr trung bình 0,73 vs 0,56, cả 4 ảnh RealVis là áo dài đúng có quần;
  SDXL base 2/4 nghiêng "áo khoác". realvis_aodai c2 đạt attr 0,86, cao nhất grid; nhìn cũng là ảnh chuẩn nhất.
- **sdxl_refplus (IP-Adapter Plus, 3 ảnh) kéo bố cục ảnh tham chiếu:** ảnh Commons là ảnh NHÓM nữ sinh → 3/4 ảnh sinh có 3–4
  người dù prompt "một cô gái"; attr cao (0,76–0,89) nhưng ITM attr thấp (0,54–0,76). Sửa: scale 0,5 → 0,4 và xếp ảnh tham chiếu
  theo P(thực thể) + độ khớp prompt thay vì chỉ P(thực thể).
- **Playground bạc màu, mờ sương ở cả 4 ảnh** — hồi quy do tôi thay VAE fp16-fix ở v1.2.1: VAE của Playground v2.5 mang
  `latents_mean/std` riêng, VAE khác không giải mã đúng. Hoàn lại VAE repo (fp32 khi giải mã, ~9,5 GB).
- **Cột PickScore trống** trong bảng: bộ chấm không nạp hoặc lỗi khi chấm, nhưng thông điệp chỉ ra stdout của cell nên không
  thấy trong HTML. Sửa: lỗi được ghi vào `MultiGenResult.notes` và hiện đỏ trong bảng điểm. Cần log cell bước 4 để biết lý do.
- Hires ×1,5 chạy được ở 1536 px không offload: đỉnh 13,4 GB trên T4 15 GB (sát), 230–260 s mỗi hàng 4 ảnh. Chưa so được với
  ảnh gốc vì PickScore trống → H11 chờ.
- Lỗi nhỏ: Bước 5 lặp 3 lần trong walkthrough.html (bấm lại cell) → Report thay mục cùng tiêu đề; thẻ GenSpec ghi "2 ứng viên"
  trong khi grid 4 → Session gán n_candidates của multigen vào GenSpec.
- Thêm `ctig/progress_report.py`: báo cáo HTML tự chứa cho người hướng dẫn (ảnh gốc, bảng/biểu đồ vector, kết luận + giả thuyết
  từ research/), `grid_hires.png`.
- Kế tiếp: chạy lại p001 (kiểm PickScore, Playground, refplus), rồi p031, p050, p012; báo cáo tiến độ gom cả 4 prompt.

