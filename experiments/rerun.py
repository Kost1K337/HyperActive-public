#!/usr/bin/env python3
"""Reproduce a tracked experiment from its MLflow run.

A run records everything that determines its result: the command line, the
seeds, the code revision, the library versions and the sha256 of every input
file. This script reads a run back and checks that the current checkout can
reproduce it, then optionally re-executes the command.

What is verified before re-running:

* input files - by sha256 of their content, not by name. A pool with the same
  file name but different content is the usual reason a "reproduction" differs;
* code revision - the current git commit against the one recorded, and whether
  the recorded run was made from a dirty working tree (in which case the commit
  does not identify the code and the result is not reproducible by definition);
* library versions - torch, stable-baselines3, gymnasium and numpy decide the
  arithmetic; a mismatch is reported because it can move the last digits.

Examples::

    python experiments/rerun.py --run-id 7f3c...           # inspect and verify
    python experiments/rerun.py --run-id 7f3c... --execute # verify, then run it
    python experiments/rerun.py --last --experiment hyperactive-training --execute
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hyperactive.tracking import environment_info, sha256_of  # noqa: E402
from hyperactive.tracking.mlflow_tracking import DEFAULT_TRACKING_URI, TRACKING_URI_ENV  # noqa: E402

# Versions that change the arithmetic, and therefore the last digits of an NPV.
CRITICAL_PACKAGES = ("torch", "stable_baselines3", "gymnasium", "numpy", "python")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-id", help="MLflow run id to reproduce")
    source.add_argument("--last", action="store_true",
                        help="take the most recent run of --experiment")
    p.add_argument("--experiment", default="hyperactive-training",
                   help="experiment to search with --last")
    p.add_argument("--tracking-uri", default=None,
                   help=f"MLflow tracking URI; default: ${TRACKING_URI_ENV} "
                        f"or {DEFAULT_TRACKING_URI}")
    p.add_argument("--execute", action="store_true",
                   help="run the recorded command after the checks pass")
    p.add_argument("--force", action="store_true",
                   help="execute even if the checks found mismatches")
    return p.parse_args()


def _client(uri: str):
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(uri)
    return MlflowClient(tracking_uri=uri)


def _find_run(client, args: argparse.Namespace):
    if args.run_id:
        return client.get_run(args.run_id)
    experiment = client.get_experiment_by_name(args.experiment)
    if experiment is None:
        raise SystemExit(f"no experiment {args.experiment!r} at this tracking URI")
    runs = client.search_runs([experiment.experiment_id],
                              order_by=["attributes.start_time DESC"], max_results=200)
    if not runs:
        raise SystemExit(f"experiment {args.experiment!r} has no runs")
    # A benchmark suite starts before its cells, so the most recent run is
    # normally a cell - and a cell is not a reproducible unit: the command,
    # the seeds and the inputs are recorded on the suite that owns it.
    top_level = next((run for run in runs if not run.data.tags.get("mlflow.parentRunId")), None)
    if top_level is None:
        raise SystemExit(f"experiment {args.experiment!r} has only nested runs; "
                         "pass --run-id of the run to reproduce")
    return top_level


def _datasets(params: dict[str, str]) -> dict[str, dict[str, str]]:
    """Recorded inputs: ``dataset.<name>.{filename,sha256,bytes}`` parameters.

    The name itself may contain dots - the benchmark logs one input per fund as
    ``dataset.fund.<fund>.wells.sha256`` - so the field is taken from the right
    and everything before it is the name.
    """
    out: dict[str, dict[str, str]] = {}
    for key, value in params.items():
        if not key.startswith("dataset.") or "." not in key[len("dataset."):]:
            continue
        name, field = key[len("dataset."):].rsplit(".", 1)
        out.setdefault(name, {})[field] = value
    return out


def check_inputs(datasets: dict[str, dict[str, str]], command: list[str]) -> list[str]:
    """Locate every recorded input and compare the sha256 of its content."""
    problems: list[str] = []
    # A file is looked up where the command points at it, and otherwise by the
    # recorded name under the repository root.
    candidates = [Path(token) for token in command if Path(token).suffix in
                  (".csv", ".xlsx", ".xls", ".xlsm")]
    for name, entry in sorted(datasets.items()):
        filename = entry.get("filename", "")
        expected = entry.get("sha256", "")
        if not expected:
            continue
        found = next((path for path in candidates if path.name == filename), None)
        if found is None:
            matches = list(ROOT.rglob(filename))
            found = matches[0] if matches else None
        if found is None or not found.exists():
            problems.append(f"input {name}: file {filename!r} not found locally")
            continue
        actual = sha256_of(found)
        status = "OK" if actual == expected else "DIFFERENT"
        print(f"  input {name:<34} {filename:<28} {status}")
        if actual != expected:
            problems.append(f"input {name}: {found} has sha256 {actual[:12]}, "
                            f"the run used {expected[:12]}")
    return problems


def check_environment(params: dict[str, str]) -> list[str]:
    """Compare the recorded runtime with the current one."""
    problems: list[str] = []
    current = environment_info()
    for package in CRITICAL_PACKAGES:
        recorded = params.get(f"env.{package}")
        if not recorded:
            continue
        actual = str(current.get(package, "n/a"))
        if recorded != actual:
            problems.append(f"{package}: the run used {recorded}, this environment has {actual}")
    return problems


def check_code(params: dict[str, str], tags: dict[str, str]) -> list[str]:
    problems: list[str] = []
    recorded = params.get("env.git_sha") or tags.get("code.git_sha")
    if not recorded:
        return problems
    current = environment_info().get("git_sha")
    if current and current != recorded:
        problems.append(f"code: the run was made at {recorded[:12]}, "
                        f"the checkout is at {current[:12]}")
    if (params.get("env.git_dirty") or tags.get("code.git_dirty")) == "true":
        problems.append("code: the run was made from a dirty working tree, "
                        "so its commit does not identify the code")
    return problems


def main() -> int:
    args = parse_args()
    uri = args.tracking_uri or os.environ.get(TRACKING_URI_ENV, DEFAULT_TRACKING_URI)
    client = _client(uri)
    run = _find_run(client, args)

    params: dict[str, Any] = dict(run.data.params)
    tags: dict[str, Any] = dict(run.data.tags)
    command_line = params.get("run.command", "")
    command = shlex.split(command_line) if command_line else []

    print(f"run     : {run.info.run_id}  ({tags.get('mlflow.runName', '')})")
    print(f"status  : {run.info.status}")
    print(f"code    : {params.get('env.git_sha', 'n/a')[:12]}"
          f"{' (dirty)' if params.get('env.git_dirty') == 'true' else ''}")
    print(f"python  : {params.get('env.python', 'n/a')}  torch {params.get('env.torch', 'n/a')}  "
          f"sb3 {params.get('env.stable-baselines3', params.get('env.stable_baselines3', 'n/a'))}")
    seeds = ", ".join(f"{key.split('.', 1)[1]}={value}"
                      for key, value in sorted(params.items()) if key.startswith("seed."))
    print(f"seeds   : {seeds or 'n/a'}")
    print(f"command : python {command_line or 'not recorded'}")
    print("checks:")

    datasets = _datasets(params)
    problems = check_inputs(datasets, command)
    problems += check_environment(params)
    problems += check_code(params, tags)

    # Nothing recorded is not the same as nothing wrong: a run without inputs,
    # environment or command cannot be verified at all, and saying it matches
    # would be the most misleading answer available.
    if not datasets and not command and not params.get("env.python"):
        problems.append("the run records no command, inputs or environment, "
                        "so there is nothing to verify against")

    if problems:
        print("\nmismatches that can change the result:")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("  environment, code revision and inputs all match the recorded run")

    if not args.execute:
        if command:
            print(f"\nto reproduce:\n  python {command_line}")
        return 1 if problems else 0
    if problems and not args.force:
        print("\nnot executing: fix the mismatches above or pass --force", file=sys.stderr)
        return 2
    if not command:
        print("the run has no recorded command line", file=sys.stderr)
        return 2
    # argv[0] is a script path, not an executable: run it with the current
    # interpreter rather than the one that happened to record the run.
    if command[0].endswith(".py"):
        command = [sys.executable, *command]
    print(f"\nexecuting: {shlex.join(command)}", flush=True)
    return subprocess.call(command, cwd=params.get("run.cwd") or str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
