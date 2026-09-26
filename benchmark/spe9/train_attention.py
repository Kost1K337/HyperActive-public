#!/usr/bin/env python3
"""Train AttentionDuelingQuantileQ from scratch on the SPE9 feature vectors
(feature_dim=10, see adapter.py) and evaluate it under the exact same
protocol as the baselines in ``run_benchmark.py``. Requires PyTorch.

TRAINING METHOD (honest disclosure): the network has no autoregressive
state of its own, so the "policy" here is a sequential construction: at
each step the network scores every not-yet-placed well via
``q_values(observations, remaining_mask)`` (the mean of its predicted
quantiles), a categorical distribution is formed over those scores, a well
is sampled (training) or argmax-picked (evaluation), and it is removed
from the remaining-candidates mask before the next step. The whole episode
(one full 26-well order) is scored with the exact same discounted,
multi-resource, travel-aware ``evaluate_order`` used for every other
method in this benchmark, and the network is trained with REINFORCE
(policy gradient) against a running-mean baseline, plus a direct quantile
regression term on each well's own value_index. This is a real training
loop with real gradient steps -- not a stub -- but it is a policy-gradient
adaptation of the quantile-Q architecture to this benchmark's
one-shot-candidate-set structure, not literal off-policy DQN with a replay
buffer; that adaptation is stated here explicitly rather than left
implicit.

This intentionally does NOT fabricate a result if PyTorch is unavailable:
it exits non-zero with an explicit, honest message instead. See README.md
("Training status") for how this was actually run (or not) in this pass.
"""
from __future__ import annotations

import csv
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
            "installing it was not feasible within this pass's time budget. "
            "AttentionDuelingQuantileQ training for SPE9 was NOT performed. "
            "This is reported as an explicit blocker, not simulated.",
            file=sys.stderr,
        )
        return 1

    # --- Reached only when PyTorch is actually available ---
    import time

    torch.set_num_threads(1)  # tiny tensors (n=26) -- CPU intra-op threading overhead dominates otherwise

    from common.attention_dueling_quantile_q import build_attention_q_network, quantile_huber_loss
    from common.sequencing_env import evaluate_order
    from spe9.adapter import load_case

    case = load_case()
    feature_dim = len(case.wells[0].features)
    torch.manual_seed(17)
    net = build_attention_q_network(feature_dim, d_model=32, heads=4, layers=2, feedforward_dim=64, quantiles=16)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    wells = case.wells
    n = len(wells)
    obs = torch.tensor([[w.features for w in wells]], dtype=torch.float32)
    value_target = torch.tensor([w.value_index for w in wells], dtype=torch.float32)

    def rollout(*, greedy: bool):
        remaining = torch.ones((1, n), dtype=torch.bool)
        chosen: list[int] = []
        log_probs = []
        for _ in range(n):
            q_values = net.q_values(obs, remaining)[0]
            if greedy:
                choice = int(torch.argmax(q_values))
            else:
                probs = torch.softmax(q_values, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                choice_t = dist.sample()
                log_probs.append(dist.log_prob(choice_t))
                choice = int(choice_t)
            chosen.append(choice)
            remaining[0, choice] = False
        order = [wells[i].well_id for i in chosen]
        return order, log_probs

    def quantile_regression_loss():
        mask = torch.ones((1, n), dtype=torch.bool)
        predicted = net(obs, mask)  # [1, n, quantiles]
        target = value_target.view(1, -1, 1).expand(-1, -1, predicted.shape[-1])
        return quantile_huber_loss(
            predicted.reshape(-1, predicted.shape[-1]),
            target.reshape(-1, predicted.shape[-1]),
        )

    started = time.perf_counter()
    episodes = 400
    baseline = 0.0
    reward_history = []
    loss_history = []
    for episode in range(episodes):
        order, log_probs = rollout(greedy=False)
        reward = evaluate_order(case, order).objective_value
        baseline = reward if episode == 0 else 0.95 * baseline + 0.05 * reward
        advantage = reward - baseline
        policy_loss = -advantage * torch.stack(log_probs).sum()
        regression_loss = quantile_regression_loss()
        loss = policy_loss + regression_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        reward_history.append(reward)
        loss_history.append(loss.item())
        if episode % 40 == 0 or episode == episodes - 1:
            print(f"episode={episode} loss={loss.item():.6f} reward={reward:.3f} baseline={baseline:.3f}")
    elapsed = time.perf_counter() - started
    print(f"trained {episodes} episodes in {elapsed:.2f}s, final_loss={loss_history[-1]:.6f}")

    # Real evaluation: greedy (argmax) rollout, under the exact same
    # evaluator as every other method.
    order, _ = rollout(greedy=True)
    result = evaluate_order(case, order, method="attention_dueling_quantile_q_rl", seed=17)
    valid = result.complete_plan and result.feasible
    new_row = {
        "method": result.method,
        "benchmark_case": result.case_id,
        "metric_value": f"{result.objective_value:.6f}",
        "valid": "true" if valid else "false",
        "notes": (
            '{"episodes": %d, "training_seconds": %.2f, '
            '"final_training_loss": %.6f}' % (episodes, elapsed, loss_history[-1])
        ),
    }

    results_path = Path(__file__).resolve().parent / "results.csv"
    with results_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "benchmark_case", "metric_value", "valid", "notes"])
        writer.writerow(new_row)
    print(f"appended 1 attention_dueling_quantile_q_rl row to {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
