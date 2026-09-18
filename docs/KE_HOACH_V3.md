# CTIG v3 — Contract-Guided Multi-Agent Prompt Repair

Chốt ngày 2026-09-18. Thay cho `docs/KE_HOACH_LOOP.md`.

## 1. Đổi hướng: bỏ vòng lặp nhiều vòng

Dừng nhánh loop nhiều vòng có early stopping bằng điểm MLLM. Năm lý do, đều có số đo:

| lý do | số đo |
|---|---|
| bộ chấm tự động chưa đủ tin | chấm 10/10 cho tà áo dài trên một tấm áo chẽn ngắn không có tà |
| contract vừa lái sửa vừa chấm → Goodhart | "điểm sau ≥ điểm trước" thành đồng nhất thức |
| loop chưa hơn random best-of-N | 4/4 lần thử, nhánh K bốc thăm seed ≥ L |
| giữ seed đổi prompt vẫn là sinh lại toàn ảnh | không được gọi là local editing |
| vá evaluator tốn thời gian mà chưa tạo bằng chứng | đã vá bốn lượt |

Hướng mới: **thư viện visual contract do LLM soạn và người Việt kiểm chứng, cộng một giao thức
multi-agent MỘT LƯỢT phát hiện lỗi văn hoá và sửa prompt.** Gọi là *one-step prompt repair*, không gọi
là image editing.

## 2. Claim

> Contract-Guided Multi-Agent Prompt Repair for Vietnamese Cultural Image Generation

**Ba đóng góp:** (1) 50 prompt văn hoá Việt + 50 visual contract có nguồn · (2) giao thức
Observer → Critic → Refiner · (3) đánh giá sửa lỗi định danh văn hoá trên challenge subset và full set.

**KHÔNG claim:** contract tự trích hoàn toàn lúc inference · ba agent là ba model độc lập · hệ thống
chỉnh ảnh cục bộ · tổng quát cho mọi nền văn hoá · MLLM evaluator tin cậy ngang người.

Câu phải viết đúng như vầy trong bài:

> Contracts were drafted offline from Wikipedia-grounded descriptions using an LLM and subsequently
> inspected and corrected by a Vietnamese annotator.
>
> Role-specialized agents sharing the same MLLM backbone.
>
> Curated reference images are used during the repair generation.

Lúc inference module chỉ làm **Contract Retrieval**: nhận prompt/entity, lấy contract từ JSON.

## 3. Pipeline

```
Prompt P -> Generator -> I0
              -> A1 Observer  chỉ tả pixel, nêu rõ chỗ bị che / ngoài khung
              -> A2 Critic    so với contract, chỉ viện dẫn contract_id có thật, chọn ĐÚNG 1 lỗi
              -> A3 Refiner   positive repair clause + negative terms, không đổi bố cục
              -> Generator cùng seed -> I1
```

Đúng một lần sửa. Critic không chắc thì **no-op**, trả I0, và **phải báo cáo tỉ lệ no-op**.

## 4. Luật repair clause

8–20 từ · một lỗi · thuần khẳng định · cấm `not/without/avoid/instead of` · không nhắc lại vật sai ·
đặt **ngay sau câu prompt gốc, trước phần mở rộng**.

Cấm mọi từ về khung hình: close-up, full-body, wide shot, top-down, camera angle, lighting, background,
crop, zoom. Bộ lọc phải theo CỤM chứ không theo từ đơn — bản lọc từ đơn từng giết 12/32 mệnh đề, toàn
những mệnh đề định danh nhất ("flat rice noodles", "flat bamboo strips", "flat-brimmed hat").

Khung hình cần thiết phải định nghĩa sẵn trong prompt hoặc contract, không để Refiner tự đổi.

## 5. Contract v2

Thêm `visibility` cho mỗi mục và `priority` cho mỗi thực thể.

| `visibility` | nghĩa |
|---|---|
| `must_be_visible` | không thấy = fail; **không** được loại khỏi mẫu số |
| `check_if_visible` | chỉ chấm khi bộ phận trong khung |
| `prompt_action` | hành động / bối cảnh do câu prompt yêu cầu |
| `optional` | chỉ mô tả, không chấm |

2–4 `required` · 2–3 `confusables` · 1–2 mục `must_be_visible` · không thuộc tính vi mô · không tuyệt
đối hoá nếu nguồn không chứng minh · **mô tả confusable không được chứa đặc điểm mà vật đúng cũng có**.

## 6. Sáu nhánh

| nhánh | thấy I0 | contract | ref | |
|---|---:|---:|---:|---|
| `B` | | | | prompt baseline |
| `P` | | ✓ | | contract text-only |
| `R` | | | ✓ | reference-only, không agent |
| `S` | ✓ | ✓ | ✓ | single-agent |
| `M` | ✓ | ✓ | ✓ | Observer–Critic–Refiner |
| `K` | | | | random regeneration, seed khác |

`M>B` cả hệ có ích · `M>R` agent hơn được ảnh thật · `M>S` phân vai có ích · `P>B` contract text đã đủ ·
`M` không hơn `S` thì **bỏ claim multi-agent**; không hơn `R` thì **kết luận phần lớn cải thiện do ref**.

Seed: I0 và I1 cùng seed; `B/P/R/S/M` cùng seed cuối; `K` seed khác. **Mỗi nhánh đúng một ảnh**, không
best-of-N cho `M` trong khi `B` chỉ có một. Nhánh nào sinh nhiều hơn phải báo chi phí và có
compute-matched control. "Cùng seed" chỉ để giảm nhiễu — không được gọi là local repair.

## 7. Đánh giá

Nhãn người theo cặp là **bằng chứng chính**. Contract score chỉ dùng để kích hoạt repair, debug, phân
tích lỗi, và **đo tương quan với nhãn người** — không bao giờ làm bằng chứng chính, vì agent tối ưu
thẳng vào nó.

MLLM evaluator: thứ tự lựa chọn **cố định** hoặc duyệt **mọi hoán vị**; cấm xoay theo hash đường dẫn
ảnh; cấm so hai nhánh bằng hai thứ tự khác nhau; tắt cache khi đo variance; báo tương quan với người.

Bootstrap phải lấy mẫu lại **theo cụm prompt**, không theo cặp: đo được 0,500 so với 0,333, tức lấy
theo cặp cho khoảng tin cậy hẹp giả khoảng 1,5 lần.

Ảnh nhánh R/S/M dùng IP-Adapter nên phải kiểm ngưỡng sao chép 0,88 với chính ảnh điều kiện.

## 8. Được và không được

**Được:** viết/sửa contract bằng tay · dùng LLM soạn contract · chọn SDXL/RealVis vì hợp phương pháp ·
challenge subset ở bảng chính, full set ở supplementary · dùng nhiều compute hơn baseline nếu báo rõ và
có control · chọn qualitative example dễ nhìn miễn vẫn đưa failure case.

**Không được:** gọi contract lookup là automatic extraction · chọn ảnh cuối bằng mắt rồi nói agent chọn ·
chỉ báo prompt thành công · giấu việc dùng IP-Adapter · lấy contract score làm metric cuối · nói ba model
độc lập · nói local image editing · sửa contract test sau khi xem kết quả rồi vẫn gọi là held-out.
