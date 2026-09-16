"""Tập ảnh dùng để ĐÁNH GIÁ phải rời hẳn tập ảnh mà vòng sửa được nhìn."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ctig.evaluation import content_hash, ref_split

PNG_A = bytes.fromhex("89504e470d0a1a0a") + b"A" * 64
PNG_B = bytes.fromhex("89504e470d0a1a0a") + b"B" * 64
PNG_C = bytes.fromhex("89504e470d0a1a0a") + b"C" * 64

with tempfile.TemporaryDirectory() as td:
    root = Path(td) / "refs"
    sel = root / "selected" / "S001"
    can = root / "candidates" / "S001"
    sel.mkdir(parents=True)
    can.mkdir(parents=True)
    (sel / "01.jpg").write_bytes(PNG_A)
    (sel / "02.jpg").write_bytes(PNG_B)
    # candidates: 1 ảnh trùng NỘI DUNG với selected nhưng khác TÊN, 1 ảnh lặp lại trong chính candidates, 2 ảnh mới
    (can / "c01.jpg").write_bytes(PNG_A)          # trùng 01.jpg -> phải bị loại
    (can / "c02.jpg").write_bytes(PNG_C)
    (can / "c03.jpg").write_bytes(PNG_C)          # trùng c02 -> phải bị loại
    (can / "c04.jpg").write_bytes(PNG_A + b"x")
    (can / "readme.txt").write_text("không phải ảnh")

    loop, ev = ref_split(str(root), "S001", n_eval=5)
    assert len(loop) == 2, loop
    names = [Path(p).name for p in ev]
    assert names == ["c02.jpg", "c04.jpg"], names
    hl = {content_hash(p) for p in loop}
    he = {content_hash(p) for p in ev}
    assert not (hl & he), "tập đánh giá dính ảnh mà loop đã nhìn"
    assert len(he) == len(ev), "tập đánh giá có ảnh lặp"
    print("tách ảnh: loop %d, đánh giá %d, rời nhau theo băm nội dung" % (len(loop), len(ev)))

    # giới hạn n_eval
    _, ev2 = ref_split([str(root)], "S001", n_eval=1)
    assert len(ev2) == 1 and Path(ev2[0]).name == "c02.jpg", ev2

    # prompt không có ảnh -> trả rỗng, không nổ
    assert ref_split(str(root), "S999") == ([], [])
    print("prompt không có ảnh: trả rỗng, không lỗi")

print("ĐẠT")
