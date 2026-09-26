#!/usr/bin/env python3
"""Expert-bootstrapped (behavioral cloning) + REINFORCE fine-tuning of
AttentionDuelingQuantileQ on the Egg Model feature vectors (feature_dim=8).

This is the "EBA" (expert-bootstrapped attention) ablation of
``train_attention.py``: instead of training REINFORCE from randomly
initialized weights (as ``train_attention.py`` does, and which lost to
greedy on all 12 cases with a 400-episode budget -- see README.md), this
script:

1. Generates expert demonstration trajectories from ``local_swap_search_2000``
   (the strongest baseline; see ``common/expert_bootstrap.py``) -- the same
   canonical seeds used for the reported baseline plus several
   random-start seeds for trajectory diversity.
2. Pretrains the network with a cross-entropy imitation loss on these
   (state, expert-action) pairs, using ``q_values()``'s masked output
   directly as action-selection logits.
3. Continues training from the bootstrapped weights with the same REINFORCE
   policy-gradient + quantile-regression loop as ``train_attention.py``, at
   an order-of-magnitude larger episode budget.
4. Evaluates the fine-tuned policy under the exact same ``evaluate_order``
   protocol as every baseline, and appends
   ``attention_dueling_quantile_q_bootstrap_rl`` rows to results.csv
   (``train_attention.py``'s ``attention_dueling_quantile_q_rl`` rows are
   left untouched -- this is a documented ablation, not a replacement).

Does NOT fabricate a result if PyTorch is unavailable: exits non-zero with
an explicit message instead.
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
            "BLOCKER: PyTorch is not installed in this environment. "
            "Bootstrap+RL training for the Egg Model was NOT performed.",
            file=sys.stderr,
        )
        return 1

    import random
    import time

    torch.set_num_threads(1)  # tiny tensors (n<=12) -- CPU intra-op threading overhead dominates otherwise

    from common.attention_dueling_quantile_q import build_attention_q_network, quantile_huber_loss
    from common.expert_bootstrap import build_bootstrap_dataset, generate_expert_demonstrations, pretrain_bootstrap
    from common.sequencing_env import evaluate_order
    from egg_model.adapter import load_realization_cases

    cases = load_realization_cases()
    feature_dim = len(cases[0].wells[0].features)
    torch.manual_seed(17)
    net = build_attention_q_network(feature_dim, d_model=32, heads=4, layers=2, feedforward_dim=64, quantiles=16)

    # --- Phase 1: expert bootstrap (behavioral cloning) ---
    bootstrap_started = time.perf_counter()
    demos = generate_expert_demonstrations(cases)
    dataset = build_bootstrap_dataset(demos)
    bootstrap_optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    bootstrap_epochs = 30
    bootstrap_loss_history = pretrain_bootstrap(
        net, bootstrap_optimizer, dataset, epochs=bootstrap_epochs, batch_size=32
    )
    bootstrap_elapsed = time.perf_counter() - bootstrap_started
    print(
        f"bootstrap: {len(demos)} demo trajectories, {len(dataset)} (state,action) pairs, "
        f"{bootstrap_epochs} epochs in {bootstrap_elapsed:.2f}s"
    )
    for epoch_idx in (0, bootstrap_epochs // 4, bootstrap_epochs // 2, bootstrap_epochs - 1):
        print(f"  bootstrap_epoch={epoch_idx} cross_entropy_loss={bootstrap_loss_history[epoch_idx]:.6f}")

    # --- Phase 2: REINFORCE fine-tuning from bootstrapped weights ---
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
    rng = random.Random(23)

    def rollout(case, *, greedy: bool):
        wells = case.wells
        n = len(wells)
        obs = torch.tensor([[w.features for w in wells]], dtype=torch.float32)
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

    def quantile_regression_loss(case):
        wells = case.wells
        obs = torch.tensor([[w.features for w in wells]], dtype=torch.float32)
        mask = torch.ones((1, len(wells)), dtype=torch.bool)
        predicted = net(obs, mask)
        target = torch.tensor([w.value_index for w in wells], dtype=torch.float32)
        target = target.view(1, -1, 1).expand(-1, -1, predicted.shape[-1])
        return quantile_huber_loss(
            predicted.reshape(-1, predicted.shape[-1]),
            target.reshape(-1, predicted.shape[-1]),
        )

    finetune_started = time.perf_counter()
    episodes = 3000
    baseline = 0.0
    reward_history = []
    loss_history = []
    for episode in range(episodes):
        case = rng.choice(cases)
        order, log_probs = rollout(case, greedy=False)
        reward = evaluate_order(case, order).objective_value
        baseline = reward if episode == 0 else 0.95 * baseline + 0.05 * reward
        advantage = reward - baseline
        policy_loss = -advantage * torch.stack(log_probs).sum()
        regression_loss = quantile_regression_loss(case)
        loss = policy_loss + regression_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        reward_history.append(reward)
        loss_history.append(loss.item())
        if episode % 300 == 0 or episode == episodes - 1:
            print(f"episode={episode} loss={loss.item():.6f} reward={reward:.3f} baseline={baseline:.3f}")
    finetune_elapsed = time.perf_counter() - finetune_started
    print(f"fine-tuned {episodes} episodes in {finetune_elapsed:.2f}s, final_loss={loss_history[-1]:.6f}")
    total_elapsed = bootstrap_elapsed + finetune_elapsed
    print(f"total wall-clock (bootstrap + fine-tune): {total_elapsed:.2f}s")

    # --- Evaluation: exact same evaluate_order protocol as every baseline ---
    results_path = Path(__file__).resolve().parent / "results.csv"
    new_rows = []
    for case in cases:
        order, _ = rollout(case, greedy=True)
        result = evaluate_order(case, order, method="attention_dueling_quantile_q_bootstrap_rl", seed=17)
        valid = result.complete_plan and result.feasible
        new_rows.append(
            {
                "method": result.method,
                "benchmark_case": result.case_id,
                "metric_value": f"{result.objective_value:.6f}",
                "valid": "true" if valid else "false",
                "notes": (
                    '{"bootstrap_demo_trajectories": %d, "bootstrap_pairs": %d, '
                    '"bootstrap_epochs": %d, "bootstrap_seconds": %.2f, '
                    '"finetune_episodes": %d, "finetune_seconds": %.2f, '
                    '"final_bootstrap_loss": %.6f, "final_finetune_loss": %.6f}'
                    % (
                        len(demos),
                        len(dataset),
                        bootstrap_epochs,
                        bootstrap_elapsed,
                        episodes,
                        finetune_elapsed,
                        bootstrap_loss_history[-1],
                        loss_history[-1],
                    )
                ),
            }
        )

    with results_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "benchmark_case", "metric_value", "valid", "notes"])
        writer.writerows(new_rows)
    print(f"appended {len(new_rows)} attention_dueling_quantile_q_bootstrap_rl rows to {results_path}")

    # --- Honest win/loss report vs. greedy_value_density and local_swap_search_2000 ---
    existing_rows = []
    with results_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            existing_rows.append(row)

    policy_value = {row["benchmark_case"]: float(row["metric_value"]) for row in new_rows}
    greedy_value = {
        row["benchmark_case"]: float(row["metric_value"])
        for row in existing_rows
        if row["method"] == "greedy_value_density"
    }
    local_swap_rows = [row for row in existing_rows if row["method"] == "local_swap_search_2000"]

    def tally(pairs):
        wins = losses = ties = 0
        for policy_v, other_v in pairs:
            if policy_v > other_v:
                wins += 1
            elif policy_v < other_v:
                losses += 1
            else:
                ties += 1
        return wins, losses, ties

    vs_greedy_pairs = [(policy_value[c], greedy_value[c]) for c in policy_value if c in greedy_value]
    vs_local_swap_pairs = [
        (policy_value[row["benchmark_case"]], float(row["metric_value"]))
        for row in local_swap_rows
        if row["benchmark_case"] in policy_value
    ]
    wins_g, losses_g, ties_g = tally(vs_greedy_pairs)
    wins_l, losses_l, ties_l = tally(vs_local_swap_pairs)
    print(
        f"vs greedy_value_density: wins={wins_g} losses={losses_g} ties={ties_g} n={len(vs_greedy_pairs)}"
    )
    print(
        f"vs local_swap_search_2000: wins={wins_l} losses={losses_l} ties={ties_l} n={len(vs_local_swap_pairs)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
