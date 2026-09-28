#!/usr/bin/env bash
# Reproduce the full real-pipeline benchmark suite (Egg Model, SPE9, SPE10,
# Volve, NLOG, Drogon, Norne, case_10) against the bc39 checkpoint family.
#
# All raw source data is already committed under benchmark/<name>/raw/ (see
# each benchmark's README.md for provenance/license/acquisition commands if
# you want to re-fetch from the original source instead of using the vendored
# copy). This script only (re)builds the environment, regenerates the
# HyperActive-native input tables, reruns the greedy/bc39 pipeline, and runs
# the test suite -- it does not touch git history or commit anything.
#
# Usage:
#   ./benchmark/reproduce_all.sh              # full run: env + all benchmarks + tests
#   ./benchmark/reproduce_all.sh --tests-only  # skip benchmark reruns, just run pytest
#   ./benchmark/reproduce_all.sh --skip-env    # assume .venv_real already set up
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$(pwd)"
VENV="$ROOT/.venv_real"
PY="$VENV/bin/python"

TESTS_ONLY=0
SKIP_ENV=0
for arg in "$@"; do
  case "$arg" in
    --tests-only) TESTS_ONLY=1 ;;
    --skip-env) SKIP_ENV=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 1 ;;
  esac
done

if [ "$SKIP_ENV" -eq 0 ]; then
  if [ ! -x "$PY" ]; then
    echo "== creating .venv_real =="
    python3 -m venv "$VENV"
  fi
  echo "== installing hyperactive (editable) + dev/test extras =="
  "$VENV/bin/pip" install -q -e ".[dev]"
fi

echo "== test suite =="
"$PY" -m pytest tests/ -q

if [ "$TESTS_ONLY" -eq 1 ]; then
  echo "== --tests-only: skipping benchmark reruns =="
  exit 0
fi

COMMON_FLAGS=(--start 2026-01-01 --horizon-years 20 --drilling-months 12 --drilling-crews 2 --gtm-crews 1)

run_external() {
  local name="$1"
  echo
  echo "== $name: rebuilding HyperActive input =="
  "$PY" "benchmark/$name/build_hyperactive_input.py"

  echo "== $name: greedy (NPV-only, primary) =="
  "$VENV/bin/hyperactive-plan" \
    --wells "benchmark/$name/hyperactive_input/wells.xlsx" \
    --coordinates "benchmark/$name/hyperactive_input/coordinates.xlsx" \
    --greedy-penalty off "${COMMON_FLAGS[@]}" \
    --out "benchmark/$name/plan_greedy_reproduced.csv"

  echo "== $name: bc39 (deterministic, zero-shot) =="
  "$VENV/bin/hyperactive-plan" \
    --wells "benchmark/$name/hyperactive_input/wells.xlsx" \
    --coordinates "benchmark/$name/hyperactive_input/coordinates.xlsx" \
    --model models/bc39 --episodes 1 --exploration 0 "${COMMON_FLAGS[@]}" \
    --out "benchmark/$name/plan_bc39_reproduced.csv"
}

for name in egg_model spe9 spe10 volve nlog drogon norne; do
  run_external "$name"
done

echo
echo "== case_10 (bundled dataset, all 4 fund sizes) =="
"$PY" benchmark/run_benchmark.py \
  --model models/bc39 --cases benchmark/case_10 \
  --funds 48,64,150,200 --horizons 12 --crews 2x1 \
  --planning-months 240 --episodes 1 --exploration 0 --workers 4 \
  --no-tracking --out benchmark/case_10/results_reproduced.xlsx

echo
echo "== done =="
echo "Reproduced plans/results are written alongside the originally-committed"
echo "ones (suffixed _reproduced) so a diff against benchmark/results_real_pipeline.csv"
echo "shows whether the numbers reproduce. See each benchmark/<name>/README.md"
echo "for the exact historical commands and reported figures to compare against."
