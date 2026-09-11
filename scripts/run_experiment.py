"""
Chạy một thí nghiệm trên tập dev và ghi vào research/research-state.yaml (trajectory).

    python scripts/run_experiment.py H1 --config configs/kaggle_t4x2.yaml --set review.max_iters=0 --tag baseline

Quy tắc (skill autoresearch): protocol trong research/experiments/<H>-*/protocol.md phải có TRƯỚC
khi chạy; script từ chối chạy nếu chưa có.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hypothesis", help="H1, H2, ...")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--tag", default="run")
    ap.add_argument("--dev", default=str(ROOT / "data" / "dev10.txt"))
    ap.add_argument("--summary", default=None, help="cấu hình có sẵn summary.json thì chỉ ghi trajectory, không chạy")
    a = ap.parse_args()

    protos = glob.glob(str(ROOT / "research" / "experiments" / f"{a.hypothesis}-*" / "protocol.md"))
    if not protos:
        sys.exit(f"Chưa có protocol cho {a.hypothesis}. Viết research/experiments/{a.hypothesis}-<slug>/protocol.md trước, "
                 f"commit, rồi mới chạy. (quy tắc khoá protocol)")

    ids = ",".join(l.strip() for l in Path(a.dev).read_text().splitlines() if l.strip())
    run_name = f"{a.hypothesis}-{a.tag}"
    import yaml

    cfg_raw = yaml.safe_load(Path(a.config).read_text(encoding="utf-8")) or {}
    runs_dir = Path(cfg_raw.get("runs_dir", ROOT / "runs"))

    if a.summary is None:
        cmd = [sys.executable, "-m", "ctig.cli", "batch", "--config", a.config, "--ids", ids, "--run-name", run_name]
        for s in a.set:
            cmd += ["--set", s]
        print("chạy:", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=ROOT)
        summary_path = runs_dir / run_name / "summary.json"
    else:
        summary_path = Path(a.summary)
    s = json.loads(summary_path.read_text(encoding="utf-8"))

    state_path = ROOT / "research" / "research-state.yaml"
    state = yaml.safe_load(state_path.read_text(encoding="utf-8"))
    traj = state["experiments"].setdefault("trajectory", [])
    baseline = state["experiments"].get("baseline_value")
    entry = {
        "run_id": run_name, "hypothesis": a.hypothesis, "config": f"{Path(a.config).name} {' '.join(a.set)}".strip(),
        "metric_value": round(s["mean_clip_fidelity"], 3), "pass_rate": round(s["pass_rate"], 3),
        "clip_iter0": round(s["mean_clip_fidelity_iter0"], 3), "judge": round(s["mean_judge_score"], 3),
        "delta": (round(s["mean_clip_fidelity"] - baseline, 3) if baseline is not None else None),
        "wall_time_min": round(s["wall_seconds"] / 60, 1), "n_prompts": s["n_prompts"],
        "change_summary": a.tag, "timestamp": str(date.today()),
    }
    traj.append(entry)
    state["experiments"]["total_runs"] = len(traj)
    if a.tag == "baseline" or baseline is None:
        state["experiments"]["baseline_value"] = entry["metric_value"]
    best = state["experiments"].get("best_value")
    if best is None or entry["metric_value"] > best:
        state["experiments"]["best_value"] = entry["metric_value"]
    state_path.write_text(yaml.safe_dump(state, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print("đã ghi trajectory:", json.dumps(entry, ensure_ascii=False))
    print(f"tiếp theo: viết research/experiments/{a.hypothesis}-*/analysis.md và cập nhật research/findings.md")


if __name__ == "__main__":
    main()
