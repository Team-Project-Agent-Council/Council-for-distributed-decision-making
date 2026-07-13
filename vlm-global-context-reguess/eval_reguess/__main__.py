"""CLI dispatcher: ``python -m eval_reguess <subcommand> ...``

Subcommands:
  geo, geo-spatial bias metrics (CPU)
  agents, per-agent R1/R2 metrics (CPU)
  judge, LLM-as-judge verdicts (GPU / API)
  aggregate, fold per-image judge JSONs into a summary
  report, markdown + HTML report
  all, run the full pipeline (with optional --skip-judge)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--results", type=Path, required=True,
                   help="Path to results_global_context_re_guess_*/ directory")
    p.add_argument("--gt", type=Path, required=True,
                   help="Path to georc_locations.csv (ground truth)")
    p.add_argument("--out", type=Path, required=True,
                   help="Output directory (will be created)")


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(prog="python -m eval_reguess")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_geo = sub.add_parser("geo", help="Geo-spatial bias metrics")
    _add_common(p_geo)

    p_ag = sub.add_parser("agents", help="Per-agent R1/R2 metrics")
    _add_common(p_ag)

    p_hm = sub.add_parser("heatmap", help="Geographic TPR/FP heatmap + world maps")
    _add_common(p_hm)

    p_dyn = sub.add_parser("dynamics",
                           help="Re-guess dynamics metrics (R1 vs R2 shift, constructive vs destructive)")
    _add_common(p_dyn)

    p_judge = sub.add_parser("judge", help="LLM-as-judge verdicts")
    _add_common(p_judge)
    p_judge.add_argument("--image-root", type=Path, default=None,
                         help="Root dir where images live (for future use)")
    p_judge.add_argument("--model", default=None,
                         help="Override VLM_JUDGE_LLM_MODEL")
    p_judge.add_argument("--api-base", default=None,
                         help="Override VLM_JUDGE_LLM_API_BASE")
    p_judge.add_argument("--concurrency", type=int, default=1,
                         help="Max concurrent LLM calls")
    p_judge.add_argument("--limit", type=int, default=None,
                         help="Only process first N images (smoke test)")
    p_judge.add_argument("--file-list", type=str, default=None,
                         help="File with image filenames or image_ids (one per line); "
                              "only those images are judged (used by SLURM launchers)")

    p_agg = sub.add_parser("aggregate", help="Aggregate per-image judge JSONs")
    p_agg.add_argument("--out", type=Path, required=True)

    p_rep = sub.add_parser("report", help="Markdown + HTML report")
    p_rep.add_argument("--out", type=Path, required=True)

    p_all = sub.add_parser("all", help="Run full pipeline")
    _add_common(p_all)
    p_all.add_argument("--image-root", type=Path, default=None)
    p_all.add_argument("--model", default=None)
    p_all.add_argument("--api-base", default=None)
    p_all.add_argument("--concurrency", type=int, default=1)
    p_all.add_argument("--skip-judge", action="store_true",
                       help="Skip Stage 2 (no GPU needed)")
    p_all.add_argument("--no-html", action="store_true",
                       help="Skip HTML report rendering")

    args = parser.parse_args(argv)

    if args.cmd == "geo":
        from eval_reguess.geo import run as geo_run
        geo_run(args.results, args.gt, args.out)

    elif args.cmd == "agents":
        from eval_reguess.agents import run as agents_run
        agents_run(args.results, args.gt, args.out)

    elif args.cmd == "heatmap":
        from eval_reguess.heatmap import run as heatmap_run
        heatmap_run(args.results, args.gt, args.out)

    elif args.cmd == "dynamics":
        from vlm_council.analyze_rounds_re_guess import compute_dynamics
        compute_dynamics(args.results, args.gt, args.out)

    elif args.cmd == "judge":
        from eval_reguess.judge import run as judge_run
        judge_run(
            results=args.results,
            gt=args.gt,
            out=args.out,
            image_root=getattr(args, "image_root", None),
            model=args.model,
            api_base=args.api_base,
            concurrency=args.concurrency,
            limit=args.limit,
            file_list=Path(args.file_list) if args.file_list else None,
        )

    elif args.cmd == "aggregate":
        from eval_reguess.judge_aggregate import run as agg_run
        agg_run(args.out)

    elif args.cmd == "report":
        from eval_reguess.report import run as rep_run
        rep_run(args.out)

    elif args.cmd == "all":
        from eval_reguess.geo import run as geo_run
        from eval_reguess.agents import run as agents_run
        from eval_reguess.report import run as rep_run

        args.out.mkdir(parents=True, exist_ok=True)

        print("=== Stage 1: geo ===")
        geo_run(args.results, args.gt, args.out)

        print("=== Stage 1: agents ===")
        agents_run(args.results, args.gt, args.out)

        print("=== Stage 1: heatmap ===")
        from eval_reguess.heatmap import run as heatmap_run
        heatmap_run(args.results, args.gt, args.out)

        print("=== Stage 1: dynamics ===")
        from vlm_council.analyze_rounds_re_guess import compute_dynamics
        compute_dynamics(args.results, args.gt, args.out)

        if not args.skip_judge:
            from eval_reguess.judge import run as judge_run
            from eval_reguess.judge_aggregate import run as agg_run
            print("=== Stage 2: judge ===")
            judge_run(
                results=args.results,
                gt=args.gt,
                out=args.out,
                image_root=getattr(args, "image_root", None),
                model=getattr(args, "model", None),
                api_base=getattr(args, "api_base", None),
                concurrency=getattr(args, "concurrency", 1),
                limit=None,
            )
            print("=== Stage 2: aggregate ===")
            agg_run(args.out)

        print("=== Report ===")
        rep_run(args.out)

        if getattr(args, "no_html", False):
            html_path = args.out / "report.html"
            if html_path.exists():
                html_path.unlink()
                print("[report] HTML removed (--no-html)")
    else:
        parser.error(f"unknown subcommand: {args.cmd}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
