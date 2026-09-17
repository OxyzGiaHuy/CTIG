# KẾ HOẠCH GỘP — CTIG

## 0. Nói thẳng về chữ "đảm bảo thành công"

Tôi không đảm bảo được vòng sửa sẽ thắng. Không ai làm được. Nhưng tôi đảm bảo được một thứ khác, và đó mới là thứ bạn cần: **thiết kế sao cho mọi ngả kết cục đều ra một bài viết được.** Cách làm là tách đóng góp thành ba phần, trong đó hai phần không phụ thuộc vào việc vòng sửa thắng hay thua:

| | Đóng góp | Phụ thuộc kết quả? |
|---|---|---|
| C1 | VNCulture-100: 100 prompt song ngữ + bảng thực thể + manifest ảnh + bộ nhãn người theo cặp | Không |
| C2 | Đo xem verifier mở nào nhìn được văn hoá Việt (3 con số, kèm giao thức) | Không |
| C3 | Vòng sửa có hơn best-of-N ở cùng ngân sách không | Có — và **cả hai chiều đều đăng được** |

Tôi đã kiểm mã trong repo trước khi viết. Mọi con số dưới đây hoặc lấy từ bằng chứng bạn đưa, hoặc tôi đo lại được trên máy này; chỗ nào phỏng đoán tôi ghi rõ.

---

## 1. Chẩn đoán (một đoạn)

Gốc rễ **không phải** "vòng sửa kém". Gốc rễ là **bạn chưa từng đo được nó**, vì ba lý do chồng lên nhau. Một, hàm chọn mù: AUC theo điểm tổng = 0,50, đúng bằng đoán mò, và điểm văn hoá chỉ rơi vào {0, 2, 5} nên S002 và S003 có cả 4 ảnh cùng 2,0 — hoà thì `copilot.py:544` (`if ev.overall > best.overall`, so sánh chặt) giữ ảnh sớm hơn, tức ảnh chưa sửa, và đó chính là lý do hệ thống giữ ảnh bạn bảo SAI ở 2/3 prompt. Hai, phép so sánh bị nhiễm: `run_loop` khởi động bằng chính ảnh nhánh B rồi trả argmax, nên **điểm_C ≥ điểm_B là một đồng nhất thức toán học**, không phải kết quả; đồng thời `configs/vast_arms.yaml:41` đặt `n_candidates: 1` cho A/B trong khi C giữ max của 4 — mà max-of-4 trừ trung bình của 4 seed, tính trên chính số liệu của bạn, là +0,55 / +0,11 / +0,10 (trung bình **+0,25 điểm miễn phí**), lớn hơn cả biên thắng từng quan sát (+0,35 và −0,15). Ba, ba đầu vào của trục văn hoá đều hỏng vì bug chứ không vì mô hình: `_identity_question` luôn đặt đáp án đúng ở vị trí A và chỉ hỏi một lần (trong khi `judge_ab_test.py` và `compare_pair` đều hỏi hai thứ tự để khử thiên lệch), nên `identity_p = 1,00` ở mọi ảnh là dấu hiệu **thiên lệch vị trí chưa khử**, không phải "đầu vào chết"; và `refs` cấp cho IP-Adapter (`run_loop_v2.py:104,138`) đúng bằng `refs` cấp cho bộ chấm (`copilot.py:225`), tức nhánh C đang được chấm bằng chính ảnh nó được điều kiện lên. Kết luận: đừng sửa vòng lặp, đừng tải model mới — **sửa 9 bug, dựng một phép so sánh khớp ngân sách có nhánh giả, rồi mua đúng một đợt nhãn người.**

---

## 2. Các bước

### GIAI ĐOẠN 0 — Chín bản sửa. Không cần nhãn người. Không cần cổng.

Mọi con số đo trước khi xong giai đoạn này đều phải đánh dấu VÔ HIỆU (số mục 1–3 của bạn vốn đã đo trước 6 bản sửa hôm nay rồi).

| # | Sửa gì | Bằng chứng trong mã | Vì sao chí mạng |
|---|---|---|---|
| S1 | Kéo `runs/pilotC`, `bestof4`, 24 ảnh + mọi JSON từ máy vast về repo | `find . -name loop_v2.json` trên máy này trả **rỗng**; nhãn trỏ `/workspace/...` | Một lần Destroy là mất toàn bộ dữ liệu thực nghiệm. Vài MB. |
| S2 | Tách `refs_cond` (selected/, cho IP-Adapter) khỏi `refs_judge` (candidates/, cho bộ chấm) | `run_loop_v2.py:104` dùng chung một biến; `bestofn_control.py:40,53,58` sinh với `ref_images=[]` nhưng chấm bằng selected/ | Ca DUY NHẤT vòng sửa thắng (S003: 6,1 vs 4,88) chưa loại trừ được lời giải thích "chấm bằng ảnh nó vừa chép". |
| S3 | Xáo vị trí đáp án (hoặc hỏi k thứ tự lấy trung bình) trong `_identity_question` | `opts = [entity_en] + look_alikes[:4] + [...]`, đáp án đúng luôn ở A | Trước khi sửa, bạn không có quyền nói identity_p là hằng số. Chưa sửa mà bỏ nó là bỏ một tín hiệu vì một bug 5 phút. |
| S4 | Không có ảnh thật → `culture = NaN`, loại prompt khỏi bảng | `copilot.py`: `else: ev.axes["culture"] = 10.0 * ev.identity_p` → với identity_p=1,00 thì **culture = 10,0**, cao nhất bảng | ~5 prompt thiếu refs sẽ nhảy lên đỉnh cả ba nhánh cùng lúc và kéo mọi hiệu số về 0. |
| S5 | `external_prompt` thiếu file → `raise`, không trả `""` | `ls data/culture_trip \| wc -l` = **85**, thiếu C035–C050; `run_loop_v2.py:100` `if refined:` không có else | 15/50 prompt complex hiện chạy với **B ≡ A** mà không một dòng log nào báo. Chạy nốt 15 prompt: ~35 phút máy. |
| S6 | Ghi `eval` của ảnh vòng 0 vào `rounds` | `rounds.append` chỉ chạy từ n=1; `label_score.py:37` đã tự ghi nhận giới hạn này | Không có nó thì không replay được luật chọn nào trên dữ liệu cũ. 3 dòng. |
| S7 | Ghi nhận mọi ca HOÀ; báo cáo theo cả hai quy ước tie-break | `copilot.py:544` dùng `>` chặt | Đừng chọn quy ước có lợi cho nhánh mình thích. Báo cả hai là một khoảng nhạy cảm. |
| S8 | `--ids` (nhiều prompt/tiến trình, dùng chung agent), bật `keep_loaded`, `skip_grounding` khi `render=bare` | `run_loop_v2.py:65` một prompt/lần; `:91` ép `keep_loaded=0`; grounding đã chứng minh cho ảnh **trùng md5** ở bare, tốn 22–191 s/prompt | Đây là bản vá trả lại nhiều giờ máy nhất trong toàn kế hoạch. Commit d6c5b9f đã làm đúng việc này cho `judge_ab_test`. |
| S9 | `du -sh`, `free -g`, bấm giờ 5 lần park/unpark thật, ghi vào HANDOFF | "~3 giây mỗi chiều" hiện là phép chia dung lượng cho băng thông, không phải số đo | Nếu RAM host < 60 GB thì park OOM-kill giữa một lô 12 giờ. |

**Ước lượng (phỏng đoán): 12–16 giờ công, ~1 giờ máy.**

---

### GIAI ĐOẠN 1 — Bộ chấm có nhìn ra văn hoá không? Không cần nhãn người.

Hai phép thử, chạy được ngay sau GĐ0, không tốn một phút của bạn.

**T1 — thật-vs-thật khác nền văn hoá (quan trọng nhất, chưa ai làm).**
Ghép một ảnh áo dài THẬT với một ảnh qipao/hanfu/kimono THẬT, hỏi "ảnh nào là trang phục truyền thống Việt Nam". Cả hai đều là ảnh chụp nên mọi manh mối artefact (da nhựa, tay sai, nén JPEG) bị triệt tiêu, cái còn lại **đúng là tri thức văn hoá**. Không cần sinh ảnh, ảnh đối chứng tải web trong nửa buổi, chạy được ~200 cặp (CI ±0,07). Hỏi cả hai thứ tự.

- **Cổng:** acc ≥ 0,80 → Mistral có tri thức văn hoá, giữ làm verifier CHỌN. 0,65–0,80 → dùng được nhưng yếu, đi tiếp và **con số này vào bài**. < 0,65 → hạ Mistral xuống vai "viết lời phê", chọn ảnh bằng trục prompt/thẩm mỹ + DINOv2, và con số này trở thành **kết quả chính của C2**.
- Mọi ngả đều đi tiếp. Không ngả nào làm hỏng bài.

**T2 — thật-vs-sinh (`judge_ab_test.py`, đã có mã).**
Sửa hai chỗ trước khi chạy: `entity = prompts[pid].text_en` phải đổi thành `entity_en` (hiện đang chèn CẢ CÂU PROMPT — mà ảnh sinh được tạo ra từ chính câu đó, nên bias kéo acc xuống vì lý do sai); và chuẩn hoá cả hai phía về cùng kích thước, cùng nén JPEG. Thêm một đối chứng thật-vs-thật cùng nền văn hoá: **phải ra ~0,50**, nếu ra 0,8 thì model đang đọc vết nén chứ không đọc nội dung.

- **Cổng (đọc MỘT CHIỀU):** ≤ 0,60 → kết luận được là mù. ≥ 0,75 → **không** cho phép kết luận gì về năng lực văn hoá; chỉ ghi "chưa bị loại". Mốc so: FAGER 0,97 · FineGRAIN 0,83 · VQAScore 0,47.

**Ước lượng: 6–8 giờ công, 2–4 giờ máy.**

---

### GIAI ĐOẠN 2 — Lô chính, khớp ngân sách, có nhánh giả. Không cần nhãn người.

Đây là chỗ sửa lỗi thiết kế nặng nhất. **Năm nhánh, cùng seed, cùng ngân sách 4 ảnh cho MỌI nhánh**, không nhánh nào dùng IP-Adapter (để B→C chỉ khác đúng một biến, và để bài tái lập được mà không cần phát hành 2.343 ảnh giấy phép không rõ):

| Nhánh | Là gì | Trả lời câu hỏi nào |
|---|---|---|
| A4 | prompt gốc, 4 seed, chọn bằng verifier | mốc dưới |
| **B4** | prompt Culture-TRIP, 4 seed, không lời phê | **nhánh giả — sàn nhiễu.** Đây là baseline đúng của C, không phải B một ảnh |
| **Bp4** | prompt cộng dồn của vòng cuối C, sinh lại 4 seed, **không nhìn ảnh** | tách "phản hồi thị giác" khỏi "prompt dài hơn" |
| C4 | vòng sửa hiện tại, 4 ảnh | nhánh đề xuất |
| (tuỳ chọn) B4+ref | B4 có IP-Adapter | chỉ chạy nếu bạn muốn giữ IP-Adapter trong C |

Bắt đầu **24 prompt** (đo giờ thật trên 3 prompt đầu rồi mới mở lên 30–40 nếu máy còn). Chia **dev 9 / test 15 theo PROMPT**, ghi hash danh sách vào git **trước** khi chạy, cùng với commit hash của công thức verifier.

Ba điều bắt buộc, không thương lượng:
1. Verifier dùng để CHỌN ảnh được **chốt trước** lô này và không sửa sau.
2. DINOv2 trên candidates/ là **report-only**, cấm tuyệt đối vào vòng chọn. Đây là tín hiệu duy nhất bộ chấm không sinh ra; nạp nó vào ensemble chọn là mất thước độc lập cuối cùng.
3. Viết `scripts/report_final.py` **trước khi lô chạy xong**, không nhận tham số nào chỉnh được sau.

- **Cổng vận hành:** > 20% prompt lỗi → dừng, sửa, chạy lại. Nếu B4 và C4 cho ảnh gần như trùng nhau (DINOv2 giữa hai nhánh > 0,95 ở > 70% prompt) → lời phê không tạo khác biệt thị giác, và đó tự nó là một kết quả: ghi lại, bỏ qua giai đoạn 3 phần so C-B.

**Ước lượng: 6 giờ công, 10–16 giờ máy (phỏng đoán; số này phụ thuộc S8 có được làm hay không).**

---

### GIAI ĐOẠN 3 — MỘT đợt nhãn người duy nhất. Đây là bước DUY NHẤT cần bạn.

Tôi cố ý đặt nó **sau** lô chính, vì ba lý do: nhãn khi đó nằm đúng phân bố ảnh của mã đã sửa (nhãn cũ đo trên mã cũ, đã hết hạn); một đợt thay vì ba; và nó cho phép cùng lúc chọn thước đo (trên dev) và báo cáo kết quả (trên test).

Giao thức:
- **Ép chọn theo cặp**, bỏ hẳn bảng kiểm thuộc tính. Lý do bằng số: 45/51 = 88% thuộc tính đạt, mà ảnh S001 vòng 0 (áo hoa văn Trung Quốc) đạt **5/5 thuộc tính** trong khi bạn chấm tổng thể là SAI. Bảng kiểm đã bão hoà, tiêu thời gian của bạn để đổi lấy không thông tin.
- Hiển thị **2–3 ảnh thật từ candidates/** (không phải từ selected/, và không phải một tấm duy nhất), hỏi "ảnh nào Việt hơn", có nút "không phân biệt được".
- 24 prompt × 4 cặp = 96 cặp (C4-vs-B4, C4-vs-Bp4, B4-vs-A4, và một cặp chọn ở chỗ các thước bất đồng) + 12 cặp lặp đảo thứ tự = **108 cặp**.
- **Thời gian thật, nói thẳng:** dữ liệu `ms` của bạn cho trung bình 26,85 s/ảnh, trung vị 20,95 s, tổng bấm giờ 322 s cho 12 ảnh. Một cặp khó hơn một ảnh. Ước **45–60 phút liền một mạch.** Không phải 15 phút. Đây là lần duy nhất tôi xin, và sau nó bài đã đủ số để viết.
- **Xin 1–2 bạn cùng lớp gán 25 cặp trùng** (~10 phút mỗi người). Đây là khoản đầu tư tỉ suất cao nhất trong cả kế hoạch: không có nó, phản biện có quyền gạch toàn bộ phần nhãn vì người gán chính là tác giả.

Cổng:
- Tự nhất quán trên 12 cặp lặp: **báo cáo, không làm cổng dừng** (12 cặp cho CI khoảng ±0,28, một cú lỡ tay không được phép giết cả hướng).
- Tỉ lệ "không phân biệt được" > 50% → các nhánh không khác nhau theo mắt người. Đó là kết quả, viết vào bài.
- Bạn bỏ cuộc trước 60 cặp → dùng đúng số đã có, ghi CI rộng, **không xin thêm**.

---

### GIAI ĐOẠN 4 — Chọn thước (trên dev), mở test một lần, viết.

So tối đa **4 ứng viên verifier** trên 9 prompt dev bằng leave-one-prompt-out, không đụng test:

| Ứng viên | Ghi chú |
|---|---|
| U1 điểm tuyệt đối hiện tại | baseline |
| U2 so cặp `compare_pair`/`pick_best` | Mã đã có, đang bị tắt. Phép đo bỏ nó (S012: 1,34–1,64 phẳng, chọn đúng ảnh tệ nhất) chạy trên **Qwen2.5-VL-7B** — chính model trả danh sách khác biệt RỖNG HOÀN TOÀN. Kết luận cũ bị nhiễu bởi model, không phải bởi phương pháp. |
| U3 kỳ vọng trên phân bố token | Chỉ chạy nếu U2 thua. **Bắt buộc đổi thang sang A–K**: `mistral_vl.py:choice_prob` lấy `enc[0]`, nên '10' và '1' dồn chung một cột và kỳ vọng sai im lặng. |
| U4 DINOv2 trên candidates/ | **report-only**, không vào vòng chọn |

- **Cổng chọn thước:** cận dưới CI 90% bootstrap **theo cụm prompt** (không theo cặp) không chạm 0,5. Tôi cố ý không dùng ngưỡng "AUC ≥ 0,70": với 9–15 cụm prompt, CI rộng ±0,12–0,20, nên một ngưỡng cứng như vậy là cổng đồng xu. Không ứng viên nào qua → đó là C2 dạng kết quả âm, có CulturalFrames 2506.08835 chống lưng (mọi thước tự động hiện có nằm ở 0,30–0,31, trần người-người 0,38). **Lưu ý viết bài: tuyệt đối không đặt AUC cạnh hệ số tương quan 0,38 như hai con số so được — hai thang khác nhau.**
- **Mở test đúng một lần**, bằng `report_final.py` đã commit, lệnh mở ghi vào git kèm timestamp.
- Chạy `report_final.py` **trên nhánh giả B4 trước** để xem bảng dưới giả thuyết không. Nếu B4 cũng "thắng" theo bảng của bạn thì bảng hỏng, và bạn biết điều đó trước khi nó cám dỗ bạn.

**Ước lượng: 8 giờ công cho GĐ3–4 phần kỹ thuật, 20–25 giờ viết bài.**

---

## 3. Nhãn người: bước nào cần, bước nào không

**Không cần một phút nào của bạn:** toàn bộ GĐ0 (9 bản sửa), GĐ1 (T1 và T2 — hai con số về verifier), GĐ2 (lô chính 5 nhánh), và phần kiểm tra vệ sinh của DINOv2 (nó phải xếp mọi ảnh thật trên mọi ảnh sinh; hàng nghìn cặp, miễn phí).

**Cần bạn:** đúng một lần, GĐ3, **45–60 phút**, ~108 cặp. Cộng ~10 phút của 1–2 bạn cùng lớp.

Nếu bạn chỉ chịu làm 20 phút: cắt xuống 48 cặp, dồn hết vào C4-vs-B4 (contrast chính), bỏ C4-vs-Bp4 và B4-vs-A4, ghi CI rộng vào bài. Vẫn ra bài, chỉ yếu hơn.

---

## 4. KHÔNG nên làm — kèm lý do bằng số

1. **Không tải FLUX.1-Kontext hay Qwen-Image-Edit lúc này.** AUC điểm tổng đang 0,50; bạn không có thước nào để biết nó giúp hay không. Thêm nữa, đổi model sinh giữa vòng 0 (SDXL) và vòng 1+ phá tính "cùng model nền" của cả thiết kế ba nhánh, và con số 24 GB đĩa / 35–38 GB VRAM là **phỏng đoán chưa kiểm** (cache HuggingFace không chia sẻ blob giữa hai repo, nên T5-XXL sẽ bị kéo về lần nữa). Đây là việc của bài SAU.
2. **Không bật inpaint hay Differential Diffusion.** OWL-ViT đã thất bại với "the sleeves of a Ao dai"; `ev.differences` là cụm kiểu "hull is oval instead of circular" — danh từ trích ra là "hull", "sides", không nằm trong từ vựng OWL-ViT. Và cả 2601.15286 lẫn FAGER đều nhấn mạnh **không mặt nạ**, nên thắng ở đây cũng không viện dẫn được hai bài đó. Nếu vẫn muốn biết: chạy OWL-ViT trên 20 lời chê đã có trong `runs/pilotC`, đếm bao nhiêu ra hộp — 1 giờ thay vì 8.
3. **Không báo cáo B→C bằng chính điểm Mistral.** `run_loop` khởi từ `first_image` (= ảnh B) và giữ argmax → điểm_C ≥ điểm_B với xác suất 1. Đây là đồng nhất thức, không phải kết quả. Phản biện đọc `copilot.py:518–544` là thấy trong ba phút.
4. **Không so nhánh 1 ảnh với nhánh 4 ảnh.** Max-of-4 trừ trung bình 4 seed, đo trên chính số liệu của bạn: +0,55 / +0,11 / +0,10, trung bình **+0,25**. Biên thắng từng quan sát là +0,35 và −0,15. Chênh lệch trong bảng sẽ gần như hoàn toàn là thống kê thứ tự của nhiễu.
5. **Không nạp DINOv2 vào vòng chọn.** Nó là con số duy nhất không do bộ chấm sinh ra. Câu hỏi đầu tiên của phản biện sẽ là "cho tôi xem một con số không do hệ thống các anh tự cho".
6. **Không quét 3 mức strength rồi nhận "thắng ở ít nhất một mức" trên 3 prompt.** Dưới giả thuyết null, xác suất qua cổng ≈ 0,875. Một cổng không thể trượt không phải cổng.
7. **Không đặt cổng trên 4 cặp hiện có.** Tôi đã đếm lại `docs/labels_huy.json`: S001 có 2 ảnh "đúng" và 2 ảnh "sai" → 4 cặp; S002 **không có ảnh nào "đúng"**; S003 **không có ảnh nào "sai"**. Nên AUC 0,75 = 2 cặp thắng + 2 cặp HOÀ, tức toàn bộ thông tin là **2 phép so sánh, từ một prompt**. P(≥3/4 | đồng xu) = 5/16 = 0,3125.
8. **Không kết luận identity_p là "đầu vào chết" trước khi xáo vị trí.** Sửa mất 5 phút; nếu sau khi xáo nó vẫn 1,00 thì mới có quyền bỏ, và mới đáng chi 8 giờ cho DINOv2. Hệ quả kèm theo: nhánh chặn trần `c = min(c, 10*identity_p)` **chưa từng chạy lần nào** — đừng mô tả nó trong bài như một cơ chế đang hoạt động.
9. **Không dùng thang 0–10 cho kỳ vọng token.** `choice_prob` lấy `enc[0]`; '10' có token đầu là '1'.
10. **Không xin nhãn người nhiều đợt.** Số phiên mới là thứ làm người ta bỏ cuộc, không phải số phút.
11. **Không chạy 100 prompt trước khi 24 prompt cho kết quả.** Culture-TRIP mới có 85/100 file.

---

## 5. Ba kịch bản kết cục

**A. Vòng sửa thắng rõ** — C4 hơn B4, cận dưới CI 90% > 0,5 trên 15 prompt test, nhất quán với verifier chốt và DINOv2 report-only, **và** C4 cũng hơn Bp4 (tức phản hồi thị giác đóng góp thật, không chỉ prompt dài hơn).
Bài là một *method paper*: "vòng phê bình có ích cho tính đúng văn hoá khi hàm chọn được kiểm chứng ngoại tại". Bảng chính là năm nhánh khớp ngân sách, thước chính là nhãn người so cặp, verifier và DINOv2 là phụ. C1 và C2 đứng ngang hàng ngay từ đầu bài, không xếp sau.
*Xác suất tôi ước (phỏng đoán): 15–20%.*

**B. Hoà** — CI chứa 0,5, hoặc C4 hơn B4 nhưng không hơn Bp4.
Bài là *benchmark + phân tích cơ chế*, và nó mạnh hơn vẻ ngoài vì bạn có nhánh giả và nhánh prompt-only để **quy trách nhiệm được**. Luận điểm: "với hành động sửa là sinh lại toàn ảnh, vòng lặp không mua thêm gì so với best-of-N ở cùng ngân sách — tái lập dự báo của Ma et al. 2501.09732 trên miền văn hoá, với một verifier mới"; kèm ablation của chính T2I-Copilot (bỏ Quality Evaluator chỉ mất 0,008: 0,813 → 0,805) để cho thấy kết quả của bạn **nhất quán** với bài gốc chứ không phải dị thường. Điểm khác biệt so với hôm nay: lần này phép đo đáng tin, vì đã khớp ngân sách, đã bỏ rò rỉ refs, đã có nhánh giả.
*Xác suất: 45–50%.*

**C. Thua** — cận trên < 0,5.
Bài là *resource + negative result*. Đóng góp: VNCulture-100 (100 prompt + manifest + nhãn so cặp), giao thức đánh giá văn hoá Việt, và **ba con số chưa ai báo cáo**: acc thật-vs-thật khác nền văn hoá, acc thật-vs-sinh (đặt cạnh FAGER 0,97 / VQAScore 0,47), và AUC của 4 ứng viên verifier trên nhãn người. Cộng một tiểu mục định lượng "bộ chấm sai ở đâu": đếm tỉ lệ lời chê sai sự thật trên 100 mẫu ngẫu nhiên — bạn đã có 3 ca mẫu (xẻ tà "không phổ biến ở áo dài", đòn gánh là "a single pole instead of a yoke", tre vs mây), việc còn lại chỉ là đếm. CulturalFrames 2506.08835 cho bài này chỗ đứng: trần người-người 0,38, mọi thước tự động 0,30–0,31 — đây là bài toán khó, không phải bạn làm dở.
*Xác suất: 30–35%.*

Tổng: **xác suất có bài nộp được ≈ 85–90%** (phỏng đoán). Phần 10–15% còn lại gần như toàn là rủi ro hạ tầng — máy thuê bị Destroy, dữ liệu chưa kéo về (đó là lý do S1 đứng đầu danh sách).

---

## 6. Tôi đã loại gì khỏi bốn kế hoạch gốc, và vì sao

- **Loại** nhánh "tải model chỉnh ảnh theo chỉ dẫn" (Kontext/Qwen-Image-Edit) khỏi kế hoạch này. Không phải vì hướng sai — nó là hướng có bằng chứng mạnh nhất trong tài liệu — mà vì **nó chưa đo được**: cổng nhãn người của kế hoạch đó (thắng 9/10 cặp cho nhánh null) là một cổng thiết kế ra để không thể đạt, và việc đổi model sinh giữa vòng 0 và vòng 1+ phá đúng tính chất một-biến mà cả bài dựa vào. Để dành cho bài sau, sau khi verifier đã qua GĐ1.
- **Loại** phép đo "bậc 3 vs bậc 4" chấm bằng chính verifier dùng để chọn. `argmax_V` trên một tập luôn ≥ giá trị V của mọi phần tử khác của tập đó, với xác suất 1. Con số đó mang đúng 0 bit thông tin.
- **Loại** việc nạp DINOv2 vào ensemble chọn ảnh, và loại việc tune trọng số ensemble trên nhãn người rồi công bố bằng chính nhãn đó.
- **Loại** bước "replay luật chọn trên JSON cũ, vài giây, không tốn GPU": không chạy được, vì `run_loop` không lưu eval của ảnh vòng 0 khi nó không phải best, và trên máy này không có một `loop_v2.json` nào. Thay bằng S6 (3 dòng) + chạy lại.

**Việc đầu tiên nên làm hôm nay, dưới một ngày, và ba trong số đó có thể lật ngược kế hoạch:** S1 (kéo dữ liệu về), S3 (xáo vị trí rồi chấm lại 12 ảnh — quyết định DINOv2 có cần không), S5 (chạy nốt 15 prompt Culture-TRIP), và S9 (`free -g` + bấm giờ park thật). Cộng T1 với 30 cặp thử để xem con số rơi vào đâu.