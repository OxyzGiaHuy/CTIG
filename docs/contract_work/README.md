# Bàn chỉnh contract (overfit)

Ba thứ cần để chỉnh nằm cạnh nhau ở đây:

| | là gì |
|---|---|
| `contracts.json` | **file sửa** — liên kết thẳng tới `data/contracts.json`, sửa ở đây là sửa luôn bản gốc |
| `contracts_review.md` | bản in ra ĐÚNG văn bản Critic đọc, kèm bảng soi từng mục (liên kết tới `docs/contracts_review.md`) |
| `anh/S0xx.png` | ảnh lô 32 đơn vị đã chạy: nháp I0 · B · R · T · S · M, để biết prompt nào đang hỏng ở đâu |

Sửa xong thì in lại bản duyệt rồi đọc lại trước khi chạy:

```
python3 scripts/dump_contracts.py -o docs/contracts_review.md
```

## Ba cần gạt, xếp theo sức nặng

**1. `part` — chỗ Observer đi soi.** Đây là cần gạt mạnh nhất và là thứ vừa sửa xong.

```json
{"id": "gourd_cup_on_rod",
 "description": "a gourd-shaped cup mounted on that rod",
 "part": "gourd cup",
 "source": "https://en.wikipedia.org/wiki/..."}
```

`part` đi thẳng vào câu lệnh cho Observer: *"Make sure your description says something about each of
these parts: …"*. Bỏ trống thì hệ thống tự dò tên bộ phận từ `description` bằng một từ điển cố định, và
tự dò hay trượt — trước khi sửa có **19/65 mục không ai soi**: `scallion_and_onion_on_top`,
`gourd_cup_on_rod`, `curtain_hides_operators`, `open_front_two_flaps`… Mục không ai soi là mục CHẾT:
Observer không nhắc tới, nên Critic không bao giờ bắt được lỗi ở đó, dù mục vẫn nằm trong contract.
Đã thấy hậu quả ở S012 — mệnh đề sửa bàn về mái chèo thay vì cái thân thuyền tròn.

Nay cả 65 mục đều có bộ phận (`✍` trong bản duyệt = viết tay). Nhiều bộ phận cho một mục thì ngăn bằng
`|`. Tối đa 8 bộ phận mỗi thực thể, mục xếp trước được ưu tiên.

**2. `description` — thứ Critic so.** Critic chỉ biết đúng chừng này về văn hoá Việt, không gì khác.
Viết bằng từ NHÌN THẤY ĐƯỢC (hình dạng, cách hai vật nối nhau, số lượng, màu). Đừng viết thứ máy không
kiểm được từ một tấm ảnh ("trang trọng", "truyền thống").

**3. Bỏ bớt mục.** Một mục *không phải lúc nào cũng xuất hiện* còn hại hơn là thiếu mục: Critic bắt lỗi
oan, Refiner viết mệnh đề sửa, rồi hệ thống làm hỏng một tấm ảnh vốn đã đúng. Đã dính đúng kiểu này ở
S017 (thành ảnh ghép), S030 (mất khăn rằn), S031 (thành áo choàng đỏ trơn). Mục nào không chắc thì bỏ.

## Cái cần gạt này KHÔNG chạm tới

Chỉnh contract không sửa được khung hình, không sửa được ảnh nháp, và không đổi được nhánh B (B không
đọc contract). Nhánh R cũng không đọc contract — R là IP-Adapter trên ảnh thật. Nếu soi ảnh mà thấy R
đã đúng còn M vẫn sai, thì vấn đề không nằm ở contract.

## Về việc overfit

Chỉnh contract theo đúng 16 prompt này rồi báo cáo số đo trên chính 16 prompt đó là **tự chấm bài
mình**. Muốn con số còn giá trị thì chia đôi: chỉnh trên 8 prompt, 8 prompt còn lại không đụng tới, rồi
báo cáo tách hai nhóm. Chênh lệch giữa hai nhóm chính là cái giá của việc overfit, và nói ra được con
số đó thì phản biện mất chỗ bám.
