"""Synthetic well pool generator.

The public repository ships no field data. This generator produces a pool with
the structure the planner is designed for: wells grouped into clusters (pads)
that become ready at different times over several years, a dominant
horizontal-well-with-fracturing type, and discrete-looking initial rates.
All distributions are generic and parametric.

Usage::

    python -m hyperactive.data.synthetic --out data/synthetic --seed 7
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

WELL_TYPES = ("ГС+ГРП", "ГС", "МЗС+ГРП", "ННС+ГРП", "ННС", "МЗС")
WELL_TYPE_WEIGHTS = (0.74, 0.18, 0.03, 0.025, 0.015, 0.01)
LENGTHS_M = (700.0, 1000.0, 1500.0, 2000.0)
LENGTH_WEIGHTS = (0.35, 0.5, 0.1, 0.05)


def generate_pool(
    n_clusters: int = 60,
    seed: int = 7,
    reference_date: datetime = datetime(2025, 1, 1),
    area_km: float = 60.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(wells, clusters)`` tables.

    * cluster size: ``1 + Poisson(4)`` capped at 12;
    * cluster readiness: uniform from one year before ``reference_date`` to
      about six years after it; wells of a cluster become ready within half a
      year of each other; 2% of wells have no readiness date;
    * oil rate: log-normal with median 35 t/day, rounded to 0.5 t/day;
      water cut drawn uniformly so that oil is 20-45% of liquid;
    * cluster centres: uniform in an ``area_km`` square.
    """
    rng = np.random.default_rng(seed)
    wells = []
    clusters = []
    well_index = 0
    for cluster_index in range(1, n_clusters + 1):
        cluster = f"C{cluster_index:03d}"
        size = int(min(12, 1 + rng.poisson(4.0)))
        base_offset = int(rng.integers(-365, 2200))
        field = f"F{int(rng.integers(1, 5))}"
        clusters.append({
            "cluster": cluster,
            "x": round(float(rng.uniform(0.0, area_km * 1000.0)), 1),
            "y": round(float(rng.uniform(0.0, area_km * 1000.0)), 1),
            "z": 0.0,
        })
        for _ in range(size):
            well_index += 1
            oil = float(np.clip(np.round(rng.lognormal(np.log(35.0), 0.3) * 2.0) / 2.0, 5.0, 110.0))
            oil_share = float(rng.uniform(0.2, 0.45))
            liq = round(oil / oil_share, 1)
            ready = reference_date + timedelta(days=base_offset + int(rng.integers(0, 180)))
            wells.append({
                "well": f"W{well_index:04d}",
                "cluster": cluster,
                "field": field,
                "layer": f"L{int(rng.integers(1, 4))}",
                "well_type": WELL_TYPES[int(rng.choice(len(WELL_TYPES), p=WELL_TYPE_WEIGHTS))],
                "oil_rate": oil,
                "liq_rate": liq,
                "length": LENGTHS_M[int(rng.choice(len(LENGTHS_M), p=LENGTH_WEIGHTS))],
                "purpose": "production",
                "readiness_date": None if rng.random() < 0.02 else ready.date().isoformat(),
            })
    return pd.DataFrame(wells), pd.DataFrame(clusters)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a synthetic well pool.")
    parser.add_argument("--out", default="data/synthetic")
    parser.add_argument("--clusters", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wells, clusters = generate_pool(n_clusters=args.clusters, seed=args.seed)
    wells.to_csv(out / "wells.csv", index=False)
    clusters.to_csv(out / "clusters.csv", index=False)
    print(f"{len(wells)} wells in {len(clusters)} clusters -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
