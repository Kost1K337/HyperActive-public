#!/usr/bin/env python3
"""Run the real baselines (random, greedy, local-swap search) for the Egg
Model well-sequencing benchmark and write ``results.csv``.

Usage:
    python3 benchmark/egg_model/run_benchmark.py \
        --seeds 17,29,43 --random-samples 500 --local-search-evaluations 2000 \
        --out benchmark/egg_model/results.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.sequencing_env import (
    EvaluationResult,
    evaluate_order,
    greedy_value_density_order,
    local_swap_search_order,
    random_best_order,
    write_results_csv,
)
from egg_model.adapter import load_realization_cases


def parse_seeds(text: str) -> list[int]:
    return [int(part) for part in text.replace(";", ",").split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="17,29,43,71,101")
    parser.add_argument("--random-samples", type=int, default=500)
    parser.add_argument("--local-search-evaluations", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results.csv")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    cases = load_realization_cases()
    results: list[EvaluationResult] = []

    for case in cases:
        started = time.perf_counter()
        order = greedy_value_density_order(case)
        results.append(
            evaluate_order(case, order, method="greedy_value_density", seed=0, started_at=started)
        )

        for seed in seeds:
            started = time.perf_counter()
            order, _ = random_best_order(case, seed=seed, samples=args.random_samples)
            results.append(
                evaluate_order(
                    case,
                    order,
                    method=f"random_{args.random_samples}_best",
                    seed=seed,
                    started_at=started,
                    metadata={"samples": args.random_samples},
                )
            )

        for seed in seeds:
            started = time.perf_counter()
            order, _ = local_swap_search_order(case, seed=seed, max_evaluations=args.local_search_evaluations)
            results.append(
                evaluate_order(
                    case,
                    order,
                    method=f"local_swap_search_{args.local_search_evaluations}",
                    seed=seed,
                    started_at=started,
                    metadata={
                        "evaluations": args.local_search_evaluations,
                        "initial_order": "greedy_value_density",
                    },
                )
            )

    write_results_csv(results, args.out)
    print(f"wrote {len(results)} rows to {args.out} across {len(cases)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
