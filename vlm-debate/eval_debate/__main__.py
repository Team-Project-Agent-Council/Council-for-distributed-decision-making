"""CLI entry-point for eval_debate."""

from __future__ import annotations

import argparse
from pathlib import Path


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--results", required=True, help="Path to debate results directory")
    p.add_argument("--gt", required=True, help="Path to ground-truth CSV")
    p.add_argument("--out", required=True, help="Output directory for metrics + plots")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m eval_debate",
        description="Evaluate the VLM Council Debate approach",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # geo
    p_geo = sub.add_parser("geo", help="Geo-spatial accuracy + bias metrics")
    _add_common(p_geo)

    # agents
    p_agents = sub.add_parser("agents", help="Per-agent and debate-dynamics metrics")
    _add_common(p_agents)

    # heatmap
    p_hm = sub.add_parser("heatmap", help="Geographic TPR/FP heatmap + world maps")
    _add_common(p_hm)

    # dynamics
    p_dyn = sub.add_parser("dynamics", help="Debate-dynamics metrics (convergence, GT pairing analysis)")
    _add_common(p_dyn)

    # judge
    p_judge = sub.add_parser("judge", help="Run LLM-as-judge on all images")
    _add_common(p_judge)
    p_judge.add_argument("--image-root", default=None, help="Root directory for image files")
    p_judge.add_argument("--model", default=None, help="Override VLM_JUDGE_LLM_MODEL env var")
    p_judge.add_argument("--api-base", default=None, help="Override VLM_JUDGE_LLM_API_BASE env var")
    p_judge.add_argument("--concurrency", type=int, default=8)
    p_judge.add_argument("--limit", type=int, default=None, help="Process only first N images")
    p_judge.add_argument("--file-list", type=str, default=None,
                         help="File with image filenames or image_ids (one per line); "
                              "only those images are judged (used by SLURM launchers)")

    # aggregate
    p_agg = sub.add_parser("aggregate", help="Aggregate judge JSONs into judge_summary.json")
    p_agg.add_argument("--out", required=True, help="Output directory")

    # report
    p_report = sub.add_parser("report", help="Generate markdown + HTML report")
    p_report.add_argument("--out", required=True, help="Output directory")

    # all
    p_all = sub.add_parser("all", help="Run full pipeline (geo + agents + [judge] + aggregate + report)")
    _add_common(p_all)
    p_all.add_argument("--image-root", default=None)
    p_all.add_argument("--model", default=None)
    p_all.add_argument("--api-base", default=None)
    p_all.add_argument("--concurrency", type=int, default=8)
    p_all.add_argument("--limit", type=int, default=None)
    p_all.add_argument("--skip-judge", action="store_true", help="Skip LLM judge step")

    args = parser.parse_args()

    results = Path(args.results) if hasattr(args, "results") else None
    gt = Path(args.gt) if hasattr(args, "gt") else None
    out = Path(args.out)

    if args.cmd == "geo":
        from eval_debate.geo import run
        run(results, gt, out)

    elif args.cmd == "agents":
        from eval_debate.agents import run
        run(results, gt, out)

    elif args.cmd == "heatmap":
        from eval_debate.heatmap import run
        run(results, gt, out)

    elif args.cmd == "dynamics":
        from vlm_council.analyze_rounds import compute_dynamics
        compute_dynamics(results, gt, out)

    elif args.cmd == "judge":
        import os
        if args.model:
            os.environ["VLM_JUDGE_LLM_MODEL"] = args.model
        if args.api_base:
            os.environ["VLM_JUDGE_LLM_API_BASE"] = args.api_base
        from eval_debate.judge import run
        run(
            results=results,
            gt=gt,
            out=out,
            image_root=Path(args.image_root) if args.image_root else None,
            concurrency=args.concurrency,
            limit=args.limit,
            file_list=Path(args.file_list) if args.file_list else None,
        )

    elif args.cmd == "aggregate":
        from eval_debate.judge_aggregate import run
        run(out)

    elif args.cmd == "report":
        from eval_debate.report import run
        run(out)

    elif args.cmd == "all":
        import os
        if args.model:
            os.environ["VLM_JUDGE_LLM_MODEL"] = args.model
        if args.api_base:
            os.environ["VLM_JUDGE_LLM_API_BASE"] = args.api_base

        from eval_debate.geo import run as geo_run
        from eval_debate.agents import run as agents_run

        geo_run(results, gt, out)
        agents_run(results, gt, out)

        from eval_debate.heatmap import run as heatmap_run
        heatmap_run(results, gt, out)

        from vlm_council.analyze_rounds import compute_dynamics
        compute_dynamics(results, gt, out)

        if not args.skip_judge:
            from eval_debate.judge import run as judge_run
            judge_run(
                results=results,
                gt=gt,
                out=out,
                image_root=Path(args.image_root) if args.image_root else None,
                concurrency=args.concurrency,
                limit=args.limit,
            )

        from eval_debate.judge_aggregate import run as agg_run
        agg_run(out)

        from eval_debate.report import run as report_run
        report_run(out)


if __name__ == "__main__":
    main()
