"""VQAScore (t2v_metrics, clip-flant5) cho các ảnh đã làm mù của eval_metrics — chạy trong venv riêng.

    /workspace/venv_t2v/bin/python scripts/vqascore_only.py --blind-dir /workspace/runs/metrics_10/_blind \
        --run SDXL --run FLUX.1-dev --ids S001,...,S010 --model clip-flant5-xl -o /workspace/runs/metrics_10/vqa.json

Vì sao tách venv: t2v_metrics viết cho transformers 4.x (import apply_chunking_to_forward,
find_pruneable_heads_and_indices... đã bị bỏ ở 5.x); venv chính phải giữ 5.x cho Mistral-3/Qwen2.5-VL/FLUX.
Tên ảnh là băm sha1("<run>|<pid>|<nhánh>")[:12] — dựng lại y hệt eval_metrics nên không cần bảng ánh xạ.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blind-dir", required=True); ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--ids", required=True); ap.add_argument("--prompts", default="data/prompts_simple.json")
    ap.add_argument("--model", default="clip-flant5-xl"); ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    prompts = {p["id"]: p for p in json.loads(Path(a.prompts).read_text(encoding="utf-8"))}
    import t2v_metrics
    m = t2v_metrics.VQAScore(model=a.model)
    out = {}
    for nhan in a.run:
        for pid in a.ids.split(","):
            for nh in ("A", "I0", "I1"):
                h = hashlib.sha1(f"{nhan}|{pid}|{nh}".encode()).hexdigest()[:12]
                p = Path(a.blind_dir) / f"{h}.png"
                if p.exists():
                    out[h] = float(m(images=[str(p)], texts=[prompts[pid]["text_en"]]).item())
                    print(nhan, pid, nh, round(out[h], 4), flush=True)
    Path(a.out).write_text(json.dumps({"model": a.model, "scores": out}, indent=1), encoding="utf-8")
    print("VQA_DONE", len(out))


if __name__ == "__main__":
    main()
