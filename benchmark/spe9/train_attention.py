#!/usr/bin/env python3
"""Train AttentionDuelingQuantileQ from scratch on the SPE9 feature vectors
(feature_dim=10, see adapter.py). Requires PyTorch.

This intentionally does NOT fabricate a result if PyTorch is unavailable:
it exits non-zero with an explicit, honest message instead. See README.md
("Training status") for how this was actually run (or not) in this pass.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        print(
            "BLOCKER: PyTorch is not installed in this environment "
            "(`python3 -c \"import torch\"` raises ModuleNotFoundError), and "
            "installing it was not feasible within this pass's time budget "
            "(a `pip download torch --no-deps` probe timed out after 10s "
            "resolving/downloading the package). AttentionDuelingQuantileQ "
            "training for SPE9 was NOT performed. This is reported as an "
            "explicit blocker, not simulated.",
            file=sys.stderr,
        )
        return 1

    # --- Reached only when PyTorch is actually available ---
    import time

    from common.attention_dueling_quantile_q import build_attention_q_network, quantile_huber_loss
    from spe9.adapter import load_case

    case = load_case()
    feature_dim = len(case.wells[0].features)
    net = build_attention_q_network(feature_dim, d_model=32, heads=4, layers=2, feedforward_dim=64, quantiles=16)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    n = len(case.wells)
    obs = torch.tensor([[w.features for w in case.wells]], dtype=torch.float32)
    mask = torch.ones((1, n), dtype=torch.bool)
    target_value = torch.tensor([[w.value_index for w in case.wells]], dtype=torch.float32).mean(dim=1, keepdim=True)
    target = target_value.expand(1, 16)

    started = time.perf_counter()
    losses = []
    steps = 200
    for step in range(steps):
        predicted = net(obs, mask).mean(dim=1)
        loss = quantile_huber_loss(predicted, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 20 == 0:
            print(f"step={step} loss={loss.item():.6f}")
    elapsed = time.perf_counter() - started
    print(f"trained {steps} steps in {elapsed:.2f}s, final_loss={losses[-1]:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
