# HyperActive

[![tests](https://github.com/Kost1K337/HyperActive-public/actions/workflows/tests.yml/badge.svg)](https://github.com/Kost1K337/HyperActive-public/actions/workflows/tests.yml)

Reinforcement learning for sequencing well construction: given a pool of
candidate wells, drilling/GTM crews, and time and production constraints,
choose which well to drill next, one decision at a time, to maximise the net
present value (NPV) of the resulting plan.

The core idea: cast plan construction as a sequential decision problem (a
Gymnasium environment), train a masked DQN with a C-DQN loss to pick the
next well, and compare it against a greedy baseline (always take the
candidate with the highest NPV) and, on small instances, against the
brute-force optimum over well orderings.

This repository is the algorithmic core extracted from a larger production
system: the RL environment, the model, the greedy baseline, and training /
evaluation / benchmarking scripts. Service infrastructure, deployment
plumbing, and proprietary field data are not included - see
[`docs/data-format.md`](docs/data-format.md) for the input schema and a
bundled synthetic dataset to run everything against.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

hyperactive-plan --wells benchmark/case_10/48/wells_native.xlsx \
    --coordinates benchmark/case_10/48/coordinates_native.xlsx \
    --model models/arrive39 --start 2026-07-09 --horizon-years 20 --drilling-months 12 \
    --drilling-crews 2 --gtm-crews 1 --episodes 10 --exploration 0.15 --out plan.csv
```

One benchmark cell (one fund, one crew configuration, one work window):

```bash
python benchmark/run_benchmark.py --model models/arrive39 --cases benchmark/case_10 \
    --funds 48 --horizons 12 --crews 2x1 --out runs/benchmark-cell.xlsx
```

The full benchmark grid (all four bundled funds x every crew configuration x
every work window - see [`docs/reproducing.md`](docs/reproducing.md#reproducing-the-benchmark)):

```bash
python benchmark/run_benchmark.py --model models/arrive39 --cases benchmark/case_10 \
    --out runs/benchmark.xlsx
```

`hyperactive-plan` and every experiment script record their run in MLflow -
command, seeds, input and model hashes, code revision, library versions,
metrics and artifacts - so a result can be traced back and re-executed later;
see [`docs/tracking.md`](docs/tracking.md).

## What's here

| | |
|---|---|
| [`src/hyperactive/core`](src/hyperactive/core) | Domain objects: wells, tasks, crews, plans, constraints. |
| [`src/hyperactive/planning`](src/hyperactive/planning) | Crew scheduling, production profiles, NPV, constraints - the model both the greedy planner and the RL environment build candidates against. |
| [`src/hyperactive/greedy`](src/hyperactive/greedy.py) | Greedy baseline planner. |
| [`src/hyperactive/env`](src/hyperactive/env) | The Gymnasium environment and its 39-dimensional candidate feature encoding. |
| [`src/hyperactive/models`](src/hyperactive/models) | Masked C-DQN, the permutation-equivariant Q-network, behavioural cloning. |
| [`src/hyperactive/inference`](src/hyperactive/inference) | Loading a trained policy and building plans with it. |
| [`src/hyperactive/data`](src/hyperactive/data) | Table loaders and a synthetic well-pool generator. |
| [`src/hyperactive/tracking`](src/hyperactive/tracking) | MLflow run tracking: provenance, plan metrics. |
| [`experiments/train.py`](experiments/train.py) | Train a policy on randomised well subsets and planning regimes. |
| [`experiments/evaluate.py`](experiments/evaluate.py) | Greedy vs. RL vs. exhaustive search on small instances. |
| [`experiments/rerun.py`](experiments/rerun.py) | Verify and re-execute a tracked run from its MLflow record. |
| [`deploy/mlflow`](deploy/mlflow) | A tracking server (MLflow + postgres) for runs that outlive one laptop. |
| [`tests/`](tests) | The test suite: masking, the C-DQN loss formula, greedy selection, environment invariants, behavioural cloning, a golden regression check, and an inference smoke test against the released model. |
| [`models/arrive39`](models/arrive39) | A released, trained policy with its normalisation statistics and training manifest. |
| [`benchmark/run_benchmark.py`](benchmark/run_benchmark.py) | RL-vs-greedy grid benchmark across funds, crew configurations and work windows. |
| [`benchmark/case_10`](benchmark/case_10) | Four well pools used to validate the benchmark port (see below). |

## Documentation

* [`docs/method.md`](docs/method.md) - the planning problem, the MDP, the
  39-feature candidate encoding, the Q-network architecture, and the C-DQN /
  behavioural-cloning training algorithm. Start here for the math.
* [`docs/data-format.md`](docs/data-format.md) - the well and coordinate
  table schema.
* [`docs/reproducing.md`](docs/reproducing.md) - setup, training,
  evaluation, and reproducing the benchmark against previously published
  reference numbers.
* [`docs/tracking.md`](docs/tracking.md) - MLflow tracking: what every run
  records, how to run a tracking server, and how to verify and re-execute a
  recorded run.
