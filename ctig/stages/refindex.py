"""
Kho ảnh tham chiếu đánh chỉ mục CLIP (v1.6, theo ImageRAG: kho CHUYÊN MIỀN tốt hơn LAION ngẫu nhiên; truy hồi bằng caption).

Nhóm đã có ~1.400 ảnh tham chiếu do API search tải về cho 95 prompt (Drive/Data/ref_images*.zip). Đánh chỉ mục một lần
bằng CLIP (cùng backbone với `perception.clip_model`), lúc chạy truy hồi bằng caption trong vài ms, không phụ thuộc
DuckDuckGo. Web search chỉ còn là đường lùi khi kho không có ảnh đủ khớp.

    python -m ctig.stages.refindex build --root <thư mục ảnh> --out runs/_cache/ref_index.npz [--clip openai/clip-vit-base-patch32]
    python -m ctig.stages.refindex search --index runs/_cache/ref_index.npz --query "a round woven bamboo basket boat" -k 5

File .npz: emb (float16, N×D, đã chuẩn hoá), kèm .json: paths (tương đối với root), root, clip_model, folder (thư mục con,
thường là id prompt). Ngưỡng cosine text-ảnh 0,26 lấy theo ImageRAG (mọi ảnh họ truy hồi đều > 0,26).
"""

from __future__ import annotations

import json
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class RefIndex:
    def __init__(self, emb, paths: list[str], root: str, clip_model: str = "", folders: list[str] | None = None):
        import numpy as np

        self.emb = np.asarray(emb, dtype="float32")
        self.paths = list(paths)
        self.root = Path(root)
        self.clip_model = clip_model
        self.folders = folders or [Path(p).parent.name for p in paths]

    def __len__(self) -> int:
        return len(self.paths)

    @classmethod
    def load(cls, path: str | Path) -> "RefIndex":
        import numpy as np

        path = Path(path)
        data = np.load(path)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        return cls(data["emb"], meta["paths"], meta["root"], meta.get("clip_model", ""), meta.get("folders"))

    def save(self, path: str | Path) -> Path:
        import numpy as np

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, emb=self.emb.astype("float16"))
        path.with_suffix(".json").write_text(json.dumps({"paths": self.paths, "root": str(self.root), "clip_model": self.clip_model,
                                                          "folders": self.folders, "n": len(self.paths)}, ensure_ascii=False), encoding="utf-8")
        return path

    def abs_path(self, i: int) -> str:
        return str(self.root / self.paths[i])

    def search(self, query_vec, k: int = 6, min_sim: float = 0.26, folders: list[str] | None = None) -> list[tuple[str, float]]:
        """query_vec: vector CLIP đã chuẩn hoá (text hoặc ảnh). Trả [(đường dẫn tuyệt đối, cosine)] giảm dần, cosine >= min_sim."""
        import numpy as np

        q = np.asarray(query_vec, dtype="float32").reshape(-1)
        q = q / (np.linalg.norm(q) + 1e-8)
        sims = self.emb @ q
        idx = np.argsort(-sims)
        out = []
        for i in idx:
            if sims[i] < min_sim:
                break
            if folders and self.folders[i] not in folders:
                continue
            p = self.abs_path(int(i))
            if Path(p).exists():
                out.append((p, float(sims[i])))
            if len(out) >= k:
                break
        return out


def list_images(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT and p.is_file())


def build(root: str | Path, out: str | Path, clip, batch: int = 16, log=print, max_images: int | None = None) -> RefIndex:
    """Đánh chỉ mục mọi ảnh dưới root bằng clip.image_embed (CLIPProbe). Ảnh hỏng bị bỏ và ghi log."""
    import numpy as np
    from PIL import Image

    root = Path(root)
    files = list_images(root)
    # kho của nhóm có cùng ảnh ở nhiều thư mục con (evidence_images / evidence_images_complex) -> khử theo (tên, kích thước)
    uniq, seen = [], set()
    for f in files:
        try:
            key = (f.name, f.stat().st_size)
        except OSError:
            key = (f.name, -1)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(f)
    if len(uniq) < len(files):
        log(f"[refindex] bỏ {len(files) - len(uniq)} ảnh trùng (cùng tên và kích thước)")
    files = uniq
    if max_images:
        files = files[:max_images]
    embs, paths, bad = [], [], 0
    for i, f in enumerate(files):
        try:
            img = Image.open(f).convert("RGB")
            img.thumbnail((512, 512))
            v = clip.image_embed(img)
            v = v.float().cpu().numpy() if hasattr(v, "cpu") else np.asarray(v, dtype="float32")
            embs.append(v / (np.linalg.norm(v) + 1e-8))
            paths.append(str(f.relative_to(root)))
        except Exception as exc:  # noqa: BLE001
            bad += 1
            log(f"[refindex] bỏ {f.name}: {type(exc).__name__}: {str(exc)[:60]}")
        if (i + 1) % 100 == 0:
            log(f"[refindex] {i + 1}/{len(files)}")
    idx = RefIndex(np.stack(embs) if embs else np.zeros((0, 1), dtype="float32"), paths, str(root), getattr(clip, "model_id", ""))
    idx.save(out)
    log(f"[refindex] {len(paths)} ảnh đánh chỉ mục, {bad} lỗi -> {out}")
    return idx


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Kho ảnh tham chiếu CLIP")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--root", required=True); b.add_argument("--out", required=True)
    b.add_argument("--clip", default="openai/clip-vit-base-patch32"); b.add_argument("--device", default="cuda:0"); b.add_argument("--max", type=int, default=None)
    s = sub.add_parser("search"); s.add_argument("--index", required=True); s.add_argument("--query", required=True); s.add_argument("-k", type=int, default=5)
    s.add_argument("--clip", default=None); s.add_argument("--device", default="cuda:0")
    a = ap.parse_args(argv)
    from .perception import CLIPProbe

    if a.cmd == "build":
        build(a.root, a.out, CLIPProbe(a.clip, a.device), max_images=a.max)
    else:
        idx = RefIndex.load(a.index)
        clip = CLIPProbe(a.clip or idx.clip_model or "openai/clip-vit-base-patch32", a.device)
        q = clip.text_embed([a.query])[0]
        for p, sim in idx.search(q, k=a.k):
            print(f"{sim:.3f}  {p}")


if __name__ == "__main__":
    main()
