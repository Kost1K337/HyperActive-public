#!/usr/bin/env python3
"""Expert-bootstrap (behavioral cloning) demonstration generation and
pretraining, shared by ``benchmark/egg_model/train_attention.py`` and
``benchmark/spe9/train_attention.py``.

DESIGN (honest disclosure): the "expert" is ``local_swap_search_2000``'s
output order -- the strongest baseline in each benchmark's ``results.csv``
(it already beats greedy on 58/60 (case, seed) pairs on Egg Model and 5/5
seeds on SPE9). For each benchmark case, several independent expert
rollouts are generated: the same canonical seeds already used for the
``local_swap_search_2000`` baseline (starting from the greedy order), plus
a handful of additional local-swap runs started from different random
initial orders, so the demonstration set has some trajectory diversity
rather than a single canonical order repeated per case.

Each expert order is converted into one supervised (state, action) pair per
construction step: the state is the case's full feature tensor plus a
boolean mask of not-yet-placed wells, and the action is the index (into the
case's well array) of the well the expert placed next. The network is
pretrained with cross-entropy against these targets, using
``AttentionDuelingQuantileQ.q_values()``'s masked output directly as
action-selection logits (already-placed wells are masked to a large
negative value by ``q_values()`` itself, so this needs no new head).

Requires PyTorch (imported lazily, matching the lazy-import pattern used by
``benchmark/common/attention_dueling_quantile_q.py``).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from common.sequencing_env import BenchmarkCase, local_swap_search_order


@dataclass(frozen=True)
class DemoTrajectory:
    case: BenchmarkCase
    order: list[str]
    objective_value: float
    source: str


def generate_expert_demonstrations(
    cases: Sequence[BenchmarkCase],
    *,
    canonical_seeds: Sequence[int] = (17, 29, 43, 71, 101),
    canonical_evaluations: int = 2000,
    diversity_seeds: Sequence[int] = (901, 902, 903, 904, 905),
    diversity_evaluations: int = 800,
) -> list[DemoTrajectory]:
    """One expert (local-swap-search) trajectory per (case, canonical seed)
    -- matching the already-reported ``local_swap_search_2000`` baseline
    exactly (same seeds, same evaluation budget, same greedy starting
    order) -- plus additional trajectories per case started from a
    different random initial order (``diversity_seeds``), for trajectory
    diversity beyond one canonical order per case."""
    demos: list[DemoTrajectory] = []
    for case in cases:
        well_ids = [w.well_id for w in case.wells]
        for seed in canonical_seeds:
            order, value = local_swap_search_order(case, seed=seed, max_evaluations=canonical_evaluations)
            demos.append(DemoTrajectory(case, order, value, f"local_swap_canonical_seed{seed}"))
        for seed in diversity_seeds:
            rng = random.Random(seed)
            start_order = well_ids[:]
            rng.shuffle(start_order)
            order, value = local_swap_search_order(
                case, seed=seed, max_evaluations=diversity_evaluations, initial_order=start_order
            )
            demos.append(DemoTrajectory(case, order, value, f"local_swap_random_start_seed{seed}"))
    return demos


def build_bootstrap_dataset(demos: Sequence[DemoTrajectory]):
    """Convert each expert order into per-step (obs, mask, target_index)
    supervised pairs. Returns a plain list; caller stacks into batches."""
    import torch

    rows = []
    for demo in demos:
        wells = demo.case.wells
        n = len(wells)
        index_of = {w.well_id: i for i, w in enumerate(wells)}
        obs = torch.tensor([[w.features for w in wells]], dtype=torch.float32)
        remaining = [True] * n
        for well_id in demo.order:
            target_idx = index_of[well_id]
            mask = torch.tensor([remaining], dtype=torch.bool)
            rows.append((obs, mask, target_idx))
            remaining[target_idx] = False
    return rows


def pretrain_bootstrap(net, optimizer, rows, *, epochs: int, batch_size: int = 32, rng_seed: int = 17):
    """Cross-entropy behavioral-cloning pretraining over the demonstration
    dataset. Returns the mean cross-entropy loss per epoch (the pretraining
    loss curve)."""
    import torch
    import torch.nn.functional as functional

    rng = random.Random(rng_seed)
    indices = list(range(len(rows)))
    loss_history: list[float] = []
    for _epoch in range(epochs):
        rng.shuffle(indices)
        epoch_losses = []
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            obs_batch = torch.cat([rows[i][0] for i in batch_idx], dim=0)
            mask_batch = torch.cat([rows[i][1] for i in batch_idx], dim=0)
            targets = torch.tensor([rows[i][2] for i in batch_idx], dtype=torch.long)
            logits = net.q_values(obs_batch, mask_batch)
            loss = functional.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        loss_history.append(sum(epoch_losses) / len(epoch_losses))
    return loss_history
