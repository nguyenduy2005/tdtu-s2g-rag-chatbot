"""Command-line interface for the S2G pilot evaluation workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.benchmark import (
    export_runtime_queries,
    load_json,
    validate_benchmark,
    write_verification_worksheet,
)
from src.evaluation.answer_quality import summarize_answer_reviews
from src.evaluation.evaluator import evaluate_ranking_artifact, write_s2g_evaluation
from src.evaluation.runner import run_b0, run_s2g
from src.product_s2g.io import atomic_json


DEFAULT_BENCHMARK = Path("data/evaluation/s2g_pilot_v1/queries.json")
DEFAULT_CORPUS = Path("data/corpus/canonical_corpus_v1_1/chunks.jsonl")


def _verified(args: argparse.Namespace) -> dict:
    return validate_benchmark(
        args.benchmark,
        args.corpus,
        require_human_verified=not args.allow_pending_human,
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    for name in ("validate", "export-runtime", "verification-sheet", "evaluate-b0", "evaluate-s2g", "evaluate-answers"):
        command = sub.add_parser(name)
        command.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
        command.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
        command.add_argument("--allow-pending-human", action="store_true")
    sub.choices["export-runtime"].add_argument("--output", type=Path, required=True)
    sub.choices["verification-sheet"].add_argument("--output", type=Path, required=True)
    sub.choices["evaluate-b0"].add_argument("--rankings", type=Path, required=True)
    sub.choices["evaluate-b0"].add_argument("--output", type=Path, required=True)
    sub.choices["evaluate-s2g"].add_argument("--runtime-root", type=Path, required=True)
    sub.choices["evaluate-s2g"].add_argument("--output-root", type=Path, required=True)
    sub.choices["evaluate-answers"].add_argument("--review", type=Path, required=True)
    sub.choices["evaluate-answers"].add_argument("--output", type=Path, required=True)
    command = sub.add_parser("run-b0")
    command.add_argument("--runtime-queries", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("run-s2g")
    command.add_argument("--runtime-queries", type=Path, required=True)
    command.add_argument("--output-root", type=Path, required=True)
    command.add_argument("--limit", type=int)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "validate":
        result = _verified(args)
    elif args.command == "export-runtime":
        result = _verified(args)
        export_runtime_queries(args.benchmark, args.output)
        result["runtime_output"] = str(args.output)
    elif args.command == "verification-sheet":
        result = _verified(args)
        write_verification_worksheet(args.benchmark, args.corpus, args.output)
        result["worksheet_output"] = str(args.output)
    elif args.command == "run-b0":
        result = run_b0(args.runtime_queries, args.output)
    elif args.command == "run-s2g":
        result = run_s2g(args.runtime_queries, args.output_root, limit=args.limit)
    elif args.command == "evaluate-b0":
        _verified(args)
        benchmark = load_json(args.benchmark)
        artifact = load_json(args.rankings)
        result = evaluate_ranking_artifact(benchmark, artifact)
        atomic_json(args.output, result)
    elif args.command == "evaluate-s2g":
        _verified(args)
        result = write_s2g_evaluation(
            load_json(args.benchmark), args.runtime_root, args.output_root
        )
    elif args.command == "evaluate-answers":
        _verified(args)
        benchmark = load_json(args.benchmark)
        result = summarize_answer_reviews(
            load_json(args.review), {row["id"] for row in benchmark["queries"]}
        )
        atomic_json(args.output, result)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
