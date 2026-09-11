"""
    python -m ctig.cli run p050 --config configs/kaggle_t4.yaml
    python -m ctig.cli batch --config configs/kaggle_t4.yaml --limit 10
    python -m ctig.cli batch --config configs/offline.yaml            # test không cần GPU
    python -m ctig.cli kb
Ghi đè cấu hình: --set t2i.steps=30 --set review.max_iters=1
"""

from __future__ import annotations

import argparse

from .config import Config, set_dotted
from .kb import KnowledgeBase
from .pipeline import Pipeline, load_prompts
from .schema import Prompt


def _cfg(args) -> Config:
    over: dict = {}
    for kv in args.set or []:
        k, _, v = kv.partition("=")
        set_dotted(over, k, v)
    if getattr(args, "run_name", None):
        over["run_name"] = args.run_name
    return Config.load(args.config, over)


def cmd_run(args):
    cfg = _cfg(args)
    prompts = load_prompts(cfg.prompts_path)
    if args.text:
        target = Prompt("adhoc", args.text, args.text)
    else:
        target = next((p for p in prompts if p.id == args.prompt_id), None)
        if target is None:
            raise SystemExit(f"Không có prompt {args.prompt_id!r}")
    pipe = Pipeline(cfg)
    res, summary = pipe.run_batch([target])
    print(f"\nẢnh cuối: {res[0].record.final_image_path}\nBáo cáo: {pipe.run_dir / 'report.html'}")


def cmd_batch(args):
    cfg = _cfg(args)
    prompts = load_prompts(cfg.prompts_path)
    if args.ids:
        want = set(args.ids.split(","))
        prompts = [p for p in prompts if p.id in want]
    if args.difficulty:
        prompts = [p for p in prompts if p.difficulty == args.difficulty]
    if args.limit:
        prompts = prompts[: args.limit]
    pipe = Pipeline(cfg)
    results, s = pipe.run_batch(prompts)
    rec = "n/a" if s.mean_retrieval_recall < 0 else f"{s.mean_retrieval_recall:.3f}"
    print(f"""
=== {s.run_id} ===
  prompt                    {s.n_prompts} (loại {s.n_unverifiable} spec rỗng)
  đạt cuối / vòng 0         {s.pass_rate:.3f} / {s.pass_rate_iter0:.3f}
  CLIP fidelity cuối / v0   {s.mean_clip_fidelity:.3f} / {s.mean_clip_fidelity_iter0:.3f}
  judge                     {s.mean_judge_score:.3f}
  retrieval recall          {rec}
  số vòng TB                {s.mean_iterations:.2f}
  thời gian                 {s.wall_seconds / 60:.1f} phút
Báo cáo: {pipe.run_dir / 'report.html'}
User study CSV: {pipe.run_dir / 'user_study.csv'}""")


def cmd_kb(args):
    kb = KnowledgeBase.load(Config().kb_path)
    print(f"KB {kb.version}: {len(kb.all())} thực thể")
    for e in sorted(kb.all(), key=lambda x: (x.category, x.prior_strength)):
        print(f"  {e.id:26} {e.name_vi:26} {e.category:12} {e.region:10} prior {e.prior_strength:.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ctig")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--config", default=None, help="file YAML trong configs/")
        p.add_argument("--set", action="append", help="ghi đè: key.sub=value")
        p.add_argument("--run-name", dest="run_name")

    p = sub.add_parser("run"); p.add_argument("prompt_id", nargs="?", default="p001"); p.add_argument("--text"); common(p); p.set_defaults(func=cmd_run)
    p = sub.add_parser("batch"); p.add_argument("--limit", type=int); p.add_argument("--ids"); p.add_argument("--difficulty", choices=["easy", "medium", "hard"]); common(p); p.set_defaults(func=cmd_batch)
    p = sub.add_parser("kb"); p.set_defaults(func=cmd_kb)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
