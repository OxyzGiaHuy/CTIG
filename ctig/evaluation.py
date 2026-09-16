"""Thước đo ĐỘC LẬP với bảng kiểm KB.

Vì sao tách hẳn thành module riêng: vòng sửa tối ưu vào `CulturalSpec` (must_have/must_not), nên báo cáo
chính điểm ấy là tự chấm mình. Nhánh có vòng sửa tối ưu thẳng vào bảng kiểm, nhánh không có thì không —
đo hai nhánh bằng chính bảng kiểm đó là thiên vị ngay từ thiết kế.

Module này **không import CulturalSpec** và không được phép đọc must_have/must_not. Ba thước đo ở đây đều
lấy từ nguồn mà vòng sửa không chạm tới:

  1. `ref_split`       tách ảnh thật thành tập loop được nhìn và tập CẤT RIÊNG chỉ để đánh giá;
  2. `prompt_vqa`      VQAScore hỏi thẳng theo câu prompt, không theo thuộc tính KB;
  3. `ref_similarity`  tương đồng ảnh-ảnh với tập ảnh thật cất riêng.

Nhãn người (`data/labels/*.json`) mới là thước đo chính; ba cái trên là thước đo tự động bổ trợ.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")


def content_hash(path: str | Path) -> str:
    """Băm nội dung tệp. Kho ảnh của nhóm có ảnh trùng giữa các thư mục con nên so theo tên là không đủ."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _images(folder: Path) -> list[str]:
    if not folder.is_dir():
        return []
    return [str(f) for f in sorted(folder.iterdir()) if f.suffix.lower() in IMG_EXT and f.is_file()]


def ref_split(ref_dirs: str | list[str], prompt_id: str, n_eval: int = 5) -> tuple[list[str], list[str]]:
    """(ảnh cho LOOP, ảnh CẤT RIÊNG để đánh giá) — rời nhau theo băm nội dung.

    Loop được nhìn `selected/<pid>` (điều kiện IP-Adapter và hiệu chỉnh ngưỡng). Tập đánh giá lấy từ
    `candidates/<pid>`, bỏ mọi ảnh trùng nội dung với tập loop và trùng lẫn nhau. Thứ tự theo tên tệp để
    chạy lại luôn ra cùng một tập.
    """
    dirs = [d.strip() for d in (ref_dirs.split(",") if isinstance(ref_dirs, str) else ref_dirs) if str(d).strip()]
    loop: list[str] = []
    for d in dirs:
        loop += _images(Path(d) / "selected" / prompt_id)
    seen = {content_hash(p) for p in loop}
    ev: list[str] = []
    for d in dirs:
        for p in _images(Path(d) / "candidates" / prompt_id):
            if len(ev) >= n_eval:
                break
            h = content_hash(p)
            if h in seen:
                continue                 # trùng ảnh loop đã thấy -> không dùng để đánh giá
            seen.add(h)
            ev.append(p)
    return loop, ev


def prompt_vqa(agent, image: str, prompt_en: str) -> float | None:
    """VQAScore (Lin et al. 2024) hỏi theo CHÍNH câu prompt, không theo thuộc tính KB. None nếu không đo được."""
    fn = getattr(agent, "vqa_yes", None)
    if fn is None or not prompt_en:
        return None
    return fn(f'Does this figure show "{prompt_en.strip()}"? Answer Yes or No.', image)


def ref_similarity(clip, image: str, eval_refs: list[str]) -> float | None:
    """Tương đồng ảnh-ảnh trung bình với tập ảnh thật CẤT RIÊNG. None nếu thiếu CLIP hoặc thiếu ảnh."""
    if clip is None or not eval_refs or not hasattr(clip, "image_similarity"):
        return None
    vals = []
    for r in eval_refs:
        try:
            vals.append(float(clip.image_similarity(image, r)))
        except Exception:  # noqa: BLE001
            continue
    return sum(vals) / len(vals) if vals else None
