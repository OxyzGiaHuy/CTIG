"""Bảng kết quả cuối, đo bằng thước ĐỘC LẬP với thứ vòng sửa tối ưu.

    python scripts/eval_independent.py --config configs/vast_arms.yaml \
        A=/workspace/runs/fullA B=/workspace/runs/fullB B4=/workspace/runs/fullB4 C=/workspace/runs/fullC \
        -o results.json

Vì sao phải độc lập: nhánh C chọn ảnh bằng điểm Mistral, nên báo cáo hiệu số B→C bằng chính điểm đó là hệ
thống tự chấm mình. Tệ hơn, `run_loop` khởi động từ chính ảnh của B rồi trả argmax, nên `điểm_C >= điểm_B`
là một ĐỒNG NHẤT THỨC toán học chứ không phải kết quả.

Hai thước dưới đây vòng sửa KHÔNG nhìn thấy và KHÔNG tối ưu được:

  SIM   độ giống ẢNH THẬT CẤT RIÊNG. `ref_split` (ctig/evaluation.py) chia ảnh tham chiếu làm hai tập rời
        nhau THEO BĂM NỘI DUNG: `selected/` cho IP-Adapter dùng lúc sinh, `candidates/` cất riêng chỉ để
        chấm. Dùng DINOv2 vì nó phân biệt kết cấu vải và hoa văn tốt hơn CLIP-I nhiều (CultDiff,
        arXiv 2502.08914) — đúng thứ ta cần cho áo dài và nan tre.

  CLIP  độ khớp với CÂU PROMPT GỐC, không phải câu Culture-TRIP đã nở ra. Lý do: ý định của người dùng nằm
        ở câu gốc; câu nở ra chỉ là phương tiện. Chấm nhánh B bằng chính câu nó được nở ra là thiên vị nó.

Hạn chế phải ghi vào bài: KHÔNG có nhãn người, nên chưa chứng minh được SIM bám theo phán đoán của người
về tính đúng văn hoá. CultDiff huấn luyện một thước như vậy có đối chiếu người; ta thì chưa.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ctig.config import Config, set_dotted  # noqa: E402
from ctig.evaluation import ref_split  # noqa: E402
from ctig.pipeline import load_prompts  # noqa: E402

_DINO: dict = {}


def dino_embed(paths: list[str], device: str = "cuda:0", model_id: str = "facebook/dinov2-base"):
    """Vector DINOv2 đã chuẩn hoá cho từng ảnh. Nạp lười một lần."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    if "m" not in _DINO:
        _DINO["p"] = AutoImageProcessor.from_pretrained(model_id)
        _DINO["m"] = AutoModel.from_pretrained(model_id).to(device).eval()
    ims = [Image.open(p).convert("RGB") for p in paths]
    with torch.inference_mode():
        x = _DINO["p"](images=ims, return_tensors="pt").to(device)
        out = _DINO["m"](**x).last_hidden_state[:, 0]          # token CLS
    return torch.nn.functional.normalize(out, dim=-1)


def final_image(run: str, pid: str) -> str | None:
    """Ảnh CUỐI mà một nhánh chọn. Hỗ trợ cả run walkthrough lẫn run vòng sửa."""
    lf = Path(run) / pid / "loop_v2.json"
    if lf.exists():
        d = json.loads(lf.read_text())
        if d.get("final") and Path(d["final"]).exists():
            return d["final"]
    bf = Path(run) / pid / "bestofn.json"
    if bf.exists():
        d = json.loads(bf.read_text())
        # chọn theo điểm của bộ chấm, ĐÚNG như nhánh C làm, để hai nhánh cùng luật chọn
        if d.get("best_overall") and Path(d["best_overall"]).exists():
            return d["best_overall"]
    got = [p for p in sorted(glob.glob(f"{run}/{pid}/*/*.png")) if Path(p).name != "grid.png"]
    return got[0] if got else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="NHÃN=đường_dẫn_run")
    ap.add_argument("--config", required=True)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--n-eval", type=int, default=5, help="số ảnh thật cất riêng mỗi prompt")
    ap.add_argument("--top", type=int, default=2, help="lấy trung bình top-k ảnh thật giống nhất")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args(argv)

    ov: dict = {}
    for kv in a.set:
        k, _, v = kv.partition("=")
        set_dotted(ov, k, v)
    cfg = Config.load(a.config, ov)
    arms = dict(x.split("=", 1) for x in a.arms)
    prompts = {p.id: p for p in load_prompts(cfg.prompts_path)}
    ids = [i.strip() for i in a.ids.split(",")] if a.ids else sorted(prompts)
    dev = cfg.multigen.device

    from ctig.stages.perception import CLIPProbe  # noqa: E402

    clip = CLIPProbe(cfg.perception.clip_model, cfg.perception.device)

    rows, bo_qua = [], []
    for pid in ids:
        loop_refs, eval_refs = ref_split(cfg.retrieval.ref_dir, pid, a.n_eval)
        if len(eval_refs) < 2:
            bo_qua.append((pid, f"chỉ {len(eval_refs)} ảnh thật cất riêng"))
            continue
        imgs = {k: final_image(v, pid) for k, v in arms.items()}
        if any(v is None for v in imgs.values()):
            bo_qua.append((pid, "thiếu ảnh ở nhánh " + ",".join(k for k, v in imgs.items() if v is None)))
            continue
        ref_vec = dino_embed(eval_refs, dev)
        gen_vec = dino_embed([imgs[k] for k in arms], dev)
        r = {"prompt_id": pid, "n_eval_refs": len(eval_refs)}
        for i, k in enumerate(arms):
            sims = (gen_vec[i] @ ref_vec.T).tolist()
            sims.sort(reverse=True)
            r[f"sim_{k}"] = round(sum(sims[: a.top]) / min(a.top, len(sims)), 4)
            if clip is not None:
                try:
                    r[f"clip_{k}"] = round(float(clip.similarity(imgs[k], [prompts[pid].text_en])[0]), 4)
                except Exception as exc:  # noqa: BLE001
                    print(f"    CLIP lỗi {type(exc).__name__}")
        rows.append(r)
        print(f"  {pid}  " + "  ".join(f"{k} sim {r[f'sim_{k}']:.3f}" for k in arms), flush=True)

    if not rows:
        raise SystemExit("không chấm được prompt nào")

    print(f"\n{len(rows)} prompt chấm được, {len(bo_qua)} bỏ qua")
    for pid, ly_do in bo_qua[:8]:
        print(f"   bỏ {pid}: {ly_do}")

    print(f"\n{'nhánh':6s} {'SIM ảnh thật':>14s} {'CLIP prompt gốc':>17s}")
    tb = {}
    for k in arms:
        s = [r[f"sim_{k}"] for r in rows]
        c = [r[f"clip_{k}"] for r in rows if f"clip_{k}" in r]
        tb[k] = {"sim": statistics.mean(s), "sim_sd": statistics.pstdev(s),
                 "clip": statistics.mean(c) if c else None}
        print(f"{k:6s} {tb[k]['sim']:8.4f} ±{tb[k]['sim_sd']:.3f} "
              + (f"{tb[k]['clip']:17.4f}" if tb[k]["clip"] is not None else f"{'-':>17s}"))

    # So TỪNG CẶP trên cùng prompt: đây mới là phép so đúng, vì prompt khó dễ khác nhau rất nhiều.
    ks = list(arms)
    print(f"\nSo từng cặp trên cùng prompt (SIM), n = {len(rows)}:")
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            x, y = ks[i], ks[j]
            d = [r[f"sim_{y}"] - r[f"sim_{x}"] for r in rows]
            thang = sum(1 for v in d if v > 0)
            sd = statistics.pstdev(d) or 1e-9
            print(f"  {y} − {x}: trung bình {statistics.mean(d):+.4f}  ·  {y} thắng {thang}/{len(d)} prompt"
                  f"  ·  d = {statistics.mean(d) / sd:+.2f}")

    print("\nLƯU Ý: SIM và CLIP đều KHÔNG phải thứ vòng sửa tối ưu (nó tối ưu điểm Mistral), và ảnh thật dùng"
          "\nở đây rời hẳn bộ ảnh IP-Adapter đã nhìn. Nhưng KHÔNG có nhãn người, nên chưa chứng minh được SIM"
          "\nbám theo phán đoán của người về tính đúng văn hoá — phải ghi vào phần Hạn chế.")

    if a.out:
        Path(a.out).write_text(json.dumps({"rows": rows, "tong_ket": tb, "bo_qua": bo_qua},
                                          ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
