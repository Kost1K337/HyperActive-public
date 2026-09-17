"""Behavioural cloning (BC) of the greedy policy for an SB3 DQN.

Prefilling the replay buffer with greedy episodes only puts demonstrations into
the buffer and waits for Q-learning to find them. Here supervised steps on
``(observation, mask) -> greedy action`` pairs are performed in addition, so the
Q-network ranks the greedy action first before the first Q-learning update.

``demo_noise_prob`` makes some demonstration steps take a random valid action
while the label stays greedy. This shows the cloned policy states off the greedy
trajectory together with the recovery target, so the clone survives its own
mistakes instead of falling apart after the first deviation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch as th
from torch import nn

from hyperactive.models.masked_q_network import INVALID_ACTION_FILL


@dataclass
class BehavioralCloningConfig:
    """Parameters of the cloning stage that precedes Q-learning."""

    enabled: bool = True

    # Demonstration collection.
    demo_episodes: int = 10
    demo_noise_prob: float = 0.15
    max_steps_multiplier: int = 2
    reuse_prefill_demos: bool = True
    add_demos_to_replay: bool = True
    skip_fallback_labels: bool = True

    # Optimisation.
    epochs: int = 40
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.0
    loss_type: str = "cross_entropy"
    margin: float = 1.0
    q_l2_coef: float = 1e-3
    grad_clip_norm: float = 10.0

    # Early stopping.
    val_fraction: float = 0.1
    early_stopping_patience: int = 8
    min_delta: float = 1e-4
    target_accuracy: float = 0.98
    restore_best_weights: bool = True

    sync_target: bool = True

    # Exploration after cloning. With exploration_initial_eps=1.0 the cloned
    # policy would be ignored for the whole decay window, so the warm start has
    # to be followed by a smaller initial epsilon.
    post_bc_exploration_initial_eps: Optional[float] = 0.3

    history_name: str = "behavioral_cloning_history.csv"
    log_every_epochs: int = 1


def behavioral_cloning_loss(
    q_net: nn.Module,
    obs: th.Tensor,
    action_masks: th.Tensor,
    expert_actions: th.Tensor,
    *,
    loss_type: str = "cross_entropy",
    margin: float = 1.0,
    q_l2_coef: float = 0.0,
) -> tuple[th.Tensor, dict[str, float]]:
    """Supervised loss pulling the policy towards the demonstrated actions.

    ``loss_type="cross_entropy"`` treats the Q-values of valid actions as the
    logits of a softmax policy, i.e. plain behavioural cloning::

        L = -log softmax_{a in valid}(Q(s, a))[a_E]

    ``loss_type="margin"`` is the large-margin classification loss of DQfD::

        L = max_{a in valid} [Q(s, a) + l(a_E, a)] - Q(s, a_E),  l = margin * [a != a_E]

    It only requires the demonstrated action to lead the other valid actions
    by ``margin`` and therefore distorts the Q scale less before Q-learning.

    ``q_l2_coef`` penalises the magnitude of valid Q-values. Both losses only
    constrain differences of Q-values, so without this term the absolute scale
    can drift far from the reward scale Q-learning works with.
    """

    if expert_actions.dim() != 1:
        raise ValueError(f"expert_actions must have shape [batch], got {tuple(expert_actions.shape)}")
    if action_masks.shape != obs.shape[:2]:
        raise ValueError(
            f"action_masks must have shape {tuple(obs.shape[:2])}, got {tuple(action_masks.shape)}"
        )
    if expert_actions.shape[0] != obs.shape[0]:
        raise ValueError(
            f"Batch size of expert_actions {expert_actions.shape[0]} does not match obs {obs.shape[0]}"
        )

    expert_actions = expert_actions.to(device=obs.device, dtype=th.long)
    action_masks = action_masks.to(device=obs.device, dtype=th.bool)
    expert_index = expert_actions.unsqueeze(1)

    if not action_masks.gather(1, expert_index).all():
        raise ValueError("The demonstration batch contains an action that is invalid under its mask.")

    q_values = q_net(obs, action_masks)
    masked_q = q_values.masked_fill(~action_masks, INVALID_ACTION_FILL)
    expert_q = q_values.gather(1, expert_index)

    if loss_type == "cross_entropy":
        loss = nn.functional.cross_entropy(masked_q, expert_actions)
    elif loss_type == "margin":
        margins = th.full_like(q_values, float(margin))
        margins.scatter_(1, expert_index, 0.0)
        penalized_q = (q_values + margins).masked_fill(~action_masks, INVALID_ACTION_FILL)
        best_penalized_q = penalized_q.max(dim=1, keepdim=True).values
        loss = (best_penalized_q - expert_q).clamp_min(0.0).mean()
    else:
        raise ValueError(
            f"Unknown loss_type={loss_type!r} for cloning. Use 'cross_entropy' or 'margin'."
        )

    valid = action_masks.float()
    valid_count = valid.sum().clamp_min(1.0)
    q_square_mean = (q_values.pow(2) * valid).sum() / valid_count
    if q_l2_coef > 0.0:
        loss = loss + float(q_l2_coef) * q_square_mean

    with th.no_grad():
        predicted = masked_q.argmax(dim=1)
        metrics = {
            "loss": float(loss.item()),
            "accuracy": float((predicted == expert_actions).float().mean().item()),
            "expert_q_mean": float(expert_q.mean().item()),
            "q_rms": float(q_square_mean.sqrt().item()),
        }

    return loss, metrics


def _candidate_names(env) -> list[str]:
    return list(env.env_method("get_candidate_well_names")[0])


def _action_mask(env, n_actions: int) -> np.ndarray:
    mask = np.asarray(env.env_method("get_action_mask")[0], dtype=np.bool_)
    if mask.shape != (n_actions,):
        raise RuntimeError(f"Action mask has shape {mask.shape}, expected ({n_actions},).")
    return mask


def _original_obs(env) -> np.ndarray:
    """Unnormalised observation of the last VecNormalize step."""

    getter = getattr(env, "get_original_obs", None)
    if getter is None:
        raise RuntimeError("Demonstrations require a VecNormalize environment.")
    return np.asarray(getter()[0], dtype=np.float32)


def _original_reward(env, reward_batch: np.ndarray) -> float:
    getter = getattr(env, "get_original_reward", None)
    if getter is None:
        return float(reward_batch[0])
    return float(getter()[0])


def _terminal_original_obs(env, info: dict[str, Any], fallback: np.ndarray) -> np.ndarray:
    terminal = info.get("terminal_observation")
    if terminal is None:
        return np.asarray(fallback, dtype=np.float32)
    terminal = np.asarray(terminal, dtype=np.float32)[None, ...]
    unnormalize = getattr(env, "unnormalize_obs", None)
    if unnormalize is not None:
        terminal = np.asarray(unnormalize(terminal), dtype=np.float32)
    return terminal[0]


def _choose_greedy_action(
    *,
    candidate_names: list[str],
    remaining_greedy_order: list[str],
    action_mask: np.ndarray,
) -> tuple[int, Optional[str], bool]:
    """The action leading to the earliest not yet executed well of the greedy plan.

    ``remaining_greedy_order`` is not modified: the caller removes the well that
    was actually executed, which on noisy steps is not the one returned here.
    The third return value is True when no candidate belongs to the greedy plan
    and the first valid action is used as a fallback.
    """

    position = {name: index for index, name in enumerate(remaining_greedy_order)}
    best_action: Optional[int] = None
    best_name: Optional[str] = None
    best_position = float("inf")

    for action, name in enumerate(candidate_names):
        if not action_mask[action]:
            continue
        candidate_position = position.get(name)
        if candidate_position is not None and candidate_position < best_position:
            best_action = int(action)
            best_name = name
            best_position = candidate_position

    if best_action is not None:
        return best_action, best_name, False

    valid_actions = np.flatnonzero(action_mask)
    if valid_actions.size == 0:
        raise RuntimeError("Cannot build a greedy demonstration: there are no valid actions.")
    action = int(valid_actions[0])
    name = candidate_names[action] if action < len(candidate_names) else None
    return action, name, True


def _sample_alternative_action(
    *,
    action_mask: np.ndarray,
    expert_action: int,
    rng: np.random.Generator,
) -> int:
    """A random valid action, different from the greedy one when possible."""

    valid_actions = np.flatnonzero(action_mask)
    if valid_actions.size == 0:
        raise RuntimeError("Cannot add noise to a demonstration: there are no valid actions.")
    alternatives = valid_actions[valid_actions != int(expert_action)]
    pool = alternatives if alternatives.size > 0 else valid_actions
    return int(rng.choice(pool))


def collect_greedy_demonstrations(
    *,
    env,
    greedy_order: list[str],
    n_actions: int,
    episodes: int,
    max_steps: int,
    noise_prob: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run greedy demonstration episodes through the training environment.

    Observations and rewards are stored unnormalised, exactly as the SB3 replay
    buffer stores them (it normalises at sampling time). The agent is not
    touched, so the same episodes feed both the buffer and cloning.

    Each transition holds the executed ``action`` (valid buffer data with the
    real reward) and the ``expert_action`` the greedy policy would take, which
    is the cloning label. They differ only on noisy steps.
    """

    rng = rng if rng is not None else np.random.default_rng()

    stats: dict[str, Any] = {
        "episodes": 0,
        "transitions": 0,
        "fallback_actions": 0,
        "noise_actions": 0,
        "final_npvs": [],
        "plan_sizes": [],
    }
    transitions: list[dict[str, Any]] = []

    if int(episodes) <= 0:
        return transitions, stats

    if not greedy_order:
        print("WARNING: the greedy plan is empty, no demonstrations were collected.")
        return transitions, stats

    for episode_index in range(1, int(episodes) + 1):
        env.reset()
        obs = _original_obs(env)
        action_mask = _action_mask(env, n_actions)
        remaining_order = list(greedy_order)
        done = False
        final_info: dict[str, Any] = {}
        steps = 0

        while not done and steps < max_steps:
            candidate_names = _candidate_names(env)
            expert_action, _expert_name, used_fallback = _choose_greedy_action(
                candidate_names=candidate_names,
                remaining_greedy_order=remaining_order,
                action_mask=action_mask,
            )
            if used_fallback:
                stats["fallback_actions"] += 1

            action = expert_action
            if noise_prob > 0.0 and float(rng.random()) < float(noise_prob):
                action = _sample_alternative_action(
                    action_mask=action_mask,
                    expert_action=expert_action,
                    rng=rng,
                )
            is_noisy = action != expert_action
            if is_noisy:
                stats["noise_actions"] += 1

            # Candidates are rebuilt inside step(), so the name of the executed
            # well has to be taken before the step.
            executed_name = candidate_names[action] if action < len(candidate_names) else None

            _, reward_batch, done_batch, infos = env.step(np.array([action], dtype=np.int64))
            reward = _original_reward(env, reward_batch)
            done = bool(done_batch[0])
            info = dict(infos[0])
            final_info = info

            next_obs_raw = _original_obs(env)
            if done:
                next_obs = _terminal_original_obs(env, info, next_obs_raw)
                next_action_mask = np.zeros(n_actions, dtype=np.bool_)
            else:
                next_obs = next_obs_raw
                next_action_mask = _action_mask(env, n_actions)

            transitions.append(
                {
                    "obs": obs.copy(),
                    "action_mask": action_mask.copy(),
                    "action": int(action),
                    "reward": float(reward),
                    "next_obs": next_obs.copy(),
                    "next_action_mask": next_action_mask.copy(),
                    "done": bool(done),
                    "expert_action": int(expert_action),
                    "used_fallback": bool(used_fallback),
                    "noisy_action": bool(is_noisy),
                }
            )
            stats["transitions"] += 1
            steps += 1

            if executed_name is not None and executed_name in remaining_order:
                remaining_order.remove(executed_name)

            obs = next_obs
            action_mask = next_action_mask

        if not done:
            raise RuntimeError(
                f"Demonstration episode did not finish: episode={episode_index}, "
                f"steps={steps}, max_steps={max_steps}"
            )

        final_plan = final_info.get("final_plan")
        stats["episodes"] += 1
        stats["final_npvs"].append(float(final_info.get("final_npv", np.nan)))
        stats["plan_sizes"].append(len(final_plan.well_plans) if final_plan is not None else None)

    return transitions, stats


def store_demonstrations(replay_buffer, transitions: list[dict[str, Any]]) -> int:
    """Put demonstration transitions into an SB3 replay buffer."""

    for transition in transitions:
        replay_buffer.add(
            obs=transition["obs"][None, ...],
            next_obs=transition["next_obs"][None, ...],
            action=np.array([[transition["action"]]], dtype=np.int64),
            reward=np.array([transition["reward"]], dtype=np.float32),
            done=np.array([transition["done"]], dtype=np.float32),
            infos=[
                {
                    "action_mask": transition["action_mask"],
                    "next_action_mask": transition["next_action_mask"],
                }
            ],
        )
    return len(transitions)


def _dataset_metrics(
    *,
    q_net: nn.Module,
    obs_all: th.Tensor,
    mask_all: th.Tensor,
    action_all: th.Tensor,
    indices: np.ndarray,
    batch_size: int,
    cfg: BehavioralCloningConfig,
    device,
) -> dict[str, float]:
    """Cloning metrics over a subset of indices, computed in chunks."""

    if len(indices) == 0:
        return {}

    totals: dict[str, float] = {}
    seen = 0
    step = max(1, int(batch_size))
    with th.no_grad():
        for start in range(0, len(indices), step):
            chunk = th.as_tensor(np.asarray(indices[start : start + step]), dtype=th.long)
            _, metrics = behavioral_cloning_loss(
                q_net,
                obs_all[chunk].to(device),
                mask_all[chunk].to(device),
                action_all[chunk].to(device),
                loss_type=cfg.loss_type,
                margin=cfg.margin,
                q_l2_coef=cfg.q_l2_coef,
            )
            weight = int(chunk.shape[0])
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * weight
            seen += weight

    return {key: value / max(seen, 1) for key, value in totals.items()}


def pretrain_behavioral_cloning(
    *,
    model,
    transitions: list[dict[str, Any]],
    cfg: BehavioralCloningConfig,
    normalize_obs=None,
    output_dir: Optional[Path] = None,
    rng: Optional[np.random.Generator] = None,
) -> dict[str, Any]:
    """Train the Q-network to reproduce the greedy action by supervised learning.

    This stage runs before the first Q-learning update, so the Q-network starts
    from weights that already rank the greedy action first instead of random
    ones. Observations are normalised with the current VecNormalize statistics;
    those keep changing during training, so the cloned mapping degrades
    gradually and Q-learning takes over.
    """

    started = datetime.now()
    stats: dict[str, Any] = {
        "enabled": bool(cfg.enabled),
        "trained": False,
        "skip_reason": None,
        "loss_type": str(cfg.loss_type),
        "dataset_transitions": 0,
        "skipped_fallback_transitions": 0,
        "train_transitions": 0,
        "val_transitions": 0,
        "epochs_run": 0,
        "stop_reason": None,
        "seconds": 0.0,
    }

    if not cfg.enabled:
        stats["skip_reason"] = "disabled"
        return stats

    if not transitions:
        stats["skip_reason"] = "no_demonstrations"
        print("WARNING: cloning is enabled but no demonstrations were collected; the stage is skipped.")
        return stats

    usable = [t for t in transitions if not (cfg.skip_fallback_labels and t.get("used_fallback"))]
    stats["skipped_fallback_transitions"] = len(transitions) - len(usable)
    stats["dataset_transitions"] = len(usable)

    if not usable:
        stats["skip_reason"] = "all_labels_are_fallback"
        print("WARNING: all demonstration labels are fallback actions; cloning is skipped.")
        return stats

    rng = rng if rng is not None else np.random.default_rng()

    raw_obs = np.stack([t["obs"] for t in usable]).astype(np.float32)
    if normalize_obs is not None:
        raw_obs = np.asarray(normalize_obs(raw_obs), dtype=np.float32)

    obs_all = th.as_tensor(raw_obs, dtype=th.float32)
    mask_all = th.as_tensor(np.stack([t["action_mask"] for t in usable]), dtype=th.bool)
    action_all = th.as_tensor(
        np.asarray([t["expert_action"] for t in usable], dtype=np.int64),
        dtype=th.long,
    )

    permutation = rng.permutation(len(usable))
    val_size = int(round(len(usable) * float(cfg.val_fraction))) if cfg.val_fraction > 0.0 else 0
    val_size = min(max(val_size, 0), max(len(usable) - 1, 0))
    val_indices = permutation[:val_size]
    train_indices = permutation[val_size:]
    stats["train_transitions"] = int(len(train_indices))
    stats["val_transitions"] = int(len(val_indices))

    q_net = model.q_net
    device = model.device
    # A separate optimiser: AdamW moments of this stage must not leak into the
    # Q-learning optimiser.
    optimizer = th.optim.AdamW(
        q_net.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )
    batch_size = max(1, min(int(cfg.batch_size), len(train_indices)))

    best_monitor = float("inf")
    best_state: Optional[dict[str, th.Tensor]] = None
    epochs_without_improvement = 0
    stop_reason = "max_epochs"
    history_rows: list[dict[str, Any]] = []

    def snapshot() -> dict[str, th.Tensor]:
        return {key: value.detach().cpu().clone() for key, value in q_net.state_dict().items()}

    model.policy.set_training_mode(True)
    print(
        f"Behavioral cloning start: transitions={len(usable)} "
        f"(train={len(train_indices)}, val={len(val_indices)}) | "
        f"loss_type={cfg.loss_type} | epochs={cfg.epochs} | batch_size={batch_size} | lr={cfg.lr}"
    )

    for epoch in range(1, int(cfg.epochs) + 1):
        epoch_order = rng.permutation(train_indices)
        train_totals: dict[str, float] = {}
        seen = 0

        for start in range(0, len(epoch_order), batch_size):
            chunk = th.as_tensor(np.asarray(epoch_order[start : start + batch_size]), dtype=th.long)
            loss, metrics = behavioral_cloning_loss(
                q_net,
                obs_all[chunk].to(device),
                mask_all[chunk].to(device),
                action_all[chunk].to(device),
                loss_type=cfg.loss_type,
                margin=cfg.margin,
                q_l2_coef=cfg.q_l2_coef,
            )
            if not np.isfinite(metrics["loss"]):
                raise RuntimeError(f"Non-finite cloning loss at epoch {epoch}: {metrics['loss']}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(q_net.parameters(), float(cfg.grad_clip_norm))
            optimizer.step()
            metrics["grad_norm"] = float(grad_norm)

            weight = int(chunk.shape[0])
            for key, value in metrics.items():
                train_totals[key] = train_totals.get(key, 0.0) + value * weight
            seen += weight

        train_metrics = {key: value / max(seen, 1) for key, value in train_totals.items()}
        val_metrics = _dataset_metrics(
            q_net=q_net,
            obs_all=obs_all,
            mask_all=mask_all,
            action_all=action_all,
            indices=val_indices,
            batch_size=cfg.batch_size,
            cfg=cfg,
            device=device,
        )

        monitor_loss = val_metrics.get("loss", train_metrics.get("loss", np.nan))
        # The target_accuracy stop looks at the worse split, not only at
        # validation: a small validation split reaches 1.0 long before the
        # policy really reproduces the greedy behaviour.
        monitor_accuracy = min(
            train_metrics.get("accuracy", np.nan),
            val_metrics.get("accuracy", np.inf),
        )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics.get("loss", np.nan),
                "train_accuracy": train_metrics.get("accuracy", np.nan),
                "val_loss": val_metrics.get("loss", np.nan),
                "val_accuracy": val_metrics.get("accuracy", np.nan),
                "monitor_loss": monitor_loss,
                "monitor_accuracy": monitor_accuracy,
                "expert_q_mean": train_metrics.get("expert_q_mean", np.nan),
                "q_rms": train_metrics.get("q_rms", np.nan),
                "grad_norm": train_metrics.get("grad_norm", np.nan),
            }
        )
        stats["epochs_run"] = epoch

        if cfg.log_every_epochs and epoch % int(cfg.log_every_epochs) == 0:
            val_loss_text = f"{val_metrics['loss']:.6f}" if "loss" in val_metrics else "no_val"
            val_accuracy_text = f"{val_metrics['accuracy']:.4f}" if "accuracy" in val_metrics else "no_val"
            print(
                f"bc_epoch={epoch:4d} | "
                f"train_loss={train_metrics.get('loss', np.nan):.6f} | "
                f"train_acc={train_metrics.get('accuracy', np.nan):.4f} | "
                f"val_loss={val_loss_text} | "
                f"val_acc={val_accuracy_text} | "
                f"q_rms={train_metrics.get('q_rms', np.nan):.4f}"
            )

        if np.isfinite(monitor_loss) and monitor_loss < best_monitor - float(cfg.min_delta):
            best_monitor = float(monitor_loss)
            best_state = snapshot()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            np.isfinite(monitor_accuracy)
            and cfg.target_accuracy is not None
            and monitor_accuracy >= float(cfg.target_accuracy)
        ):
            # The current weights already reproduce the greedy behaviour; keep
            # them even if an earlier epoch had a lower loss.
            best_state = snapshot()
            best_monitor = float(monitor_loss) if np.isfinite(monitor_loss) else best_monitor
            stop_reason = "target_accuracy"
            break

        if (
            cfg.early_stopping_patience
            and cfg.early_stopping_patience > 0
            and epochs_without_improvement >= int(cfg.early_stopping_patience)
        ):
            stop_reason = "early_stopping"
            break

    if cfg.restore_best_weights and best_state is not None:
        q_net.load_state_dict(best_state)

    if cfg.sync_target:
        model.q_net_target.load_state_dict(q_net.state_dict())

    final_train = _dataset_metrics(
        q_net=q_net,
        obs_all=obs_all,
        mask_all=mask_all,
        action_all=action_all,
        indices=train_indices,
        batch_size=cfg.batch_size,
        cfg=cfg,
        device=device,
    )
    final_val = _dataset_metrics(
        q_net=q_net,
        obs_all=obs_all,
        mask_all=mask_all,
        action_all=action_all,
        indices=val_indices,
        batch_size=cfg.batch_size,
        cfg=cfg,
        device=device,
    )

    stats.update(
        {
            "trained": True,
            "stop_reason": stop_reason,
            "best_monitor_loss": best_monitor if np.isfinite(best_monitor) else None,
            "final_train_loss": final_train.get("loss"),
            "final_train_accuracy": final_train.get("accuracy"),
            "final_val_loss": final_val.get("loss"),
            "final_val_accuracy": final_val.get("accuracy"),
            "final_q_rms": final_train.get("q_rms"),
            "target_synced": bool(cfg.sync_target),
            "seconds": (datetime.now() - started).total_seconds(),
        }
    )

    if output_dir is not None and history_rows:
        _save_history(history_rows, Path(output_dir), cfg)

    print(
        "Behavioral cloning done: "
        f"stop_reason={stop_reason} | epochs={stats['epochs_run']} | "
        f"train_acc={_format_optional(stats['final_train_accuracy'], '.4f')} | "
        f"val_acc={_format_optional(stats['final_val_accuracy'], '.4f')} | "
        f"train_loss={_format_optional(stats['final_train_loss'], '.6f')} | "
        f"q_rms={_format_optional(stats['final_q_rms'], '.4f')} | "
        f"seconds={stats['seconds']:.1f}"
    )
    return stats


def _format_optional(value: Any, spec: str = ".6f") -> str:
    if value is None:
        return "nan"
    number = float(value)
    return format(number, spec) if np.isfinite(number) else "nan"


def _save_history(history_rows: list[dict[str, Any]], output_dir: Path, cfg: BehavioralCloningConfig) -> None:
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)
    history = pd.DataFrame(history_rows)
    history.to_csv(output_dir / cfg.history_name, index=False)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    figure, loss_axis = plt.subplots(figsize=(10, 5))
    loss_axis.plot(history["epoch"], history["train_loss"], label="BC train loss")
    if history["val_loss"].notna().any():
        loss_axis.plot(history["epoch"], history["val_loss"], label="BC val loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")

    accuracy_axis = loss_axis.twinx()
    accuracy_axis.plot(history["epoch"], history["train_accuracy"], linestyle="--", label="BC train accuracy")
    if history["val_accuracy"].notna().any():
        accuracy_axis.plot(history["epoch"], history["val_accuracy"], linestyle="--", label="BC val accuracy")
    accuracy_axis.set_ylabel("Greedy action accuracy")
    accuracy_axis.set_ylim(0.0, 1.05)

    handles = loss_axis.get_legend_handles_labels()[0] + accuracy_axis.get_legend_handles_labels()[0]
    labels = loss_axis.get_legend_handles_labels()[1] + accuracy_axis.get_legend_handles_labels()[1]
    loss_axis.legend(handles, labels, loc="center right")
    loss_axis.set_title("Behavioral cloning on greedy demonstrations")

    figure.tight_layout()
    figure.savefig(output_dir / "behavioral_cloning.png", dpi=150)
    plt.close(figure)
