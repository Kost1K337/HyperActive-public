#!/usr/bin/env python3
"""Load recorded benchmark runs into a fresh tracking server.

Why. A tracking server that comes up empty says nothing about the model it is
meant to track: the first question anyone asks of it - "how does the released
policy actually do?" - needs a benchmark grid that takes hours to compute. The
grid in ``seed/`` was computed once with the command each run records, exported
run by run, and is replayed here so that the server has the released model's
numbers the moment it starts.

Seeded runs are the real ones, not summaries: the same parameters, metrics,
tags and input hashes, the same suite/cell nesting. Their provenance
(``run.command``, ``env.*``, ``dataset.*.sha256``, ``model.*.sha256``) is what
``experiments/rerun.py`` verifies, so a seeded cell can be checked against the
current checkout and recomputed like any other run.

Idempotent: every seeded run carries ``seed.id``, and a file whose id is
already present is skipped, so restarting the container does not duplicate
anything.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities import Dataset, DatasetInput, InputTag, Metric, Param, RunTag
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO, format="seed: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_URI = "http://127.0.0.1:5000"
SEED_DIR = Path(os.environ.get("HYPERACTIVE_SEED_DIR", Path(__file__).parent / "seed"))
# The server migrates its database on startup; seeding waits rather than racing it.
STARTUP_TIMEOUT_SECONDS = float(os.environ.get("HYPERACTIVE_SEED_TIMEOUT", "300"))
# MLflow rejects oversized batches; these are below the documented server limits.
PARAM_BATCH = 90
METRIC_BATCH = 900
TAG_BATCH = 90


def wait_for_server(client: MlflowClient, timeout: float) -> bool:
    deadline = time.time() + timeout
    last: Exception | None = None
    while time.time() < deadline:
        try:
            client.search_experiments(max_results=1)
            return True
        except Exception as exc:
            last = exc
            time.sleep(3)
    logger.error("tracking server did not become available: %s", last)
    return False


def already_seeded(client: MlflowClient, experiment_id: str, seed_id: str) -> bool:
    found = client.search_runs([experiment_id], filter_string=f"tags.`seed.id` = '{seed_id}'",
                               max_results=1)
    return bool(found)


def _batched(items: list[Any], size: int) -> Any:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def create_run(client: MlflowClient, experiment_id: str, spec: dict[str, Any],
               seed_id: str, parent_id: str | None = None) -> str:
    tags = dict(spec.get("tags") or {})
    tags["seed.id"] = seed_id
    tags["seed.replayed"] = "true"
    if parent_id:
        tags["mlflow.parentRunId"] = parent_id

    name = spec.get("name", "run")
    run = client.create_run(experiment_id, start_time=spec.get("start_time"),
                            tags={}, run_name=name)
    run_id = run.info.run_id

    for chunk in _batched([RunTag(k, str(v)[:5000]) for k, v in tags.items()], TAG_BATCH):
        client.log_batch(run_id, tags=chunk)
    for chunk in _batched([Param(k, str(v)) for k, v in (spec.get("params") or {}).items()],
                          PARAM_BATCH):
        client.log_batch(run_id, params=chunk)
    metrics = [Metric(m["key"], float(m["value"]), int(m["timestamp"]), int(m.get("step", 0)))
               for m in (spec.get("metrics") or [])]
    for chunk in _batched(metrics, METRIC_BATCH):
        client.log_batch(run_id, metrics=chunk)

    inputs = []
    for item in spec.get("datasets") or []:
        dataset = Dataset(name=item["name"], digest=item["digest"],
                          source_type=item.get("source_type", "local"),
                          source=item.get("source", "{}"))
        context = item.get("context")
        tags_in = [InputTag("mlflow.data.context", context)] if context else []
        inputs.append(DatasetInput(dataset=dataset, tags=tags_in))
    if inputs:
        client.log_inputs(run_id, inputs)

    for name, payload in (spec.get("artifacts") or {}).items():
        # Small JSON attachments only (economics, the flat cell table, the
        # environment): the model and the result workbook live in the
        # repository, and copying them into every seeded run buys nothing.
        client.log_text(run_id, json.dumps(payload, ensure_ascii=False, indent=2), name)

    client.set_terminated(run_id, spec.get("status", "FINISHED"), end_time=spec.get("end_time"))
    return run_id


def seed_file(client: MlflowClient, path: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    seed_id = payload["seed_id"]
    experiment_name = payload.get("experiment", "hyperactive-benchmarks")

    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = (experiment.experiment_id if experiment
                     else client.create_experiment(experiment_name))

    if already_seeded(client, experiment_id, seed_id):
        logger.info("%s: already present, skipping", seed_id)
        return False

    suite = payload["suite"]
    suite_id = create_run(client, experiment_id, suite, seed_id)
    for cell in suite.get("cells") or []:
        create_run(client, experiment_id, cell, seed_id, parent_id=suite_id)
    logger.info("%s: %d cells under %s", seed_id, len(suite.get("cells") or []), suite["name"])
    return True


def main() -> int:
    uri = os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_URI
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    logger.info("tracking server %s", uri)
    if not wait_for_server(client, STARTUP_TIMEOUT_SECONDS):
        return 1

    files = sorted(SEED_DIR.glob("*.json")) if SEED_DIR.is_dir() else []
    if not files:
        logger.info("nothing to seed in %s", SEED_DIR)
        return 0
    for path in files:
        try:
            seed_file(client, path)
        except Exception as exc:
            # A broken seed file must not stop the server from being usable.
            logger.error("%s: not seeded: %s", path.name, exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
