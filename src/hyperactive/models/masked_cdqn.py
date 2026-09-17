"""Main learning algorithm of the planner: SB3 DQN with action masking, BC and C-DQN.

Differences from ``stable_baselines3.DQN``:

* **Action masking.** ``PlanEnv`` exposes a boolean mask of valid candidates.
  Invalid actions are excluded from action selection, from the Double-DQN
  action selection and from bootstrapping, and the mask is stored in the replay
  buffer together with the transition.
* **Behavioural cloning (BC).** Before the first Q-learning step the Q-network
  can be trained with supervision to reproduce the greedy policy (see
  :mod:`hyperactive.models.behavioral_cloning`).
* **C-DQN loss** (Wang & Ueda, ICLR 2022): instead of ``l_DQN`` the agent
  minimises ``E[max(l_DQN, l_MSBE)]``. The extra term ``l_MSBE`` bootstraps
  through the online network with the gradient enabled; it upper-bounds the
  next DQN loss and makes the sequence of minima non-increasing, which removes
  divergence of Q.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.dqn import DQN
from torch.nn import functional as F

try:  # stable-baselines3 >= 2.8
    from stable_baselines3.common.utils import LinearSchedule as _LinearSchedule

    def _linear_schedule(start: float, end: float, end_fraction: float):
        return _LinearSchedule(start, end, end_fraction)

except ImportError:  # stable-baselines3 2.7.x
    from stable_baselines3.common.utils import get_linear_fn as _linear_schedule

from hyperactive.models.behavioral_cloning import (
    BehavioralCloningConfig,
    collect_greedy_demonstrations,
    pretrain_behavioral_cloning,
    store_demonstrations,
)
from hyperactive.models.masked_q_network import MaskedDQNPolicy, MaskedMlpPolicy
from hyperactive.models.masked_replay_buffer import MaskedReplayBuffer


class MaskedCDQN(DQN):
    """stable-baselines3 DQN with action masking, BC warm start and the C-DQN loss."""

    policy_aliases: ClassVar[dict[str, type]] = {
        "MlpPolicy": MaskedMlpPolicy,
        "MaskedMlpPolicy": MaskedMlpPolicy,
        "MaskedDQNPolicy": MaskedDQNPolicy,
    }

    def __init__(
        self,
        policy: Union[str, type] = "MlpPolicy",
        env: Optional[GymEnv] = None,
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 100_000,
        learning_starts: int = 1_000,
        batch_size: int = 128,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[type] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 1_000,
        exploration_fraction: float = 0.8,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10.0,
        use_cdqn: bool = True,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        # C-DQN is on by default; use_cdqn=False gives a plain masked Double
        # DQN and serves as the control arm.
        self.use_cdqn = bool(use_cdqn)
        self.last_update_info: dict[str, float] = {}
        self._last_action_masks: Optional[np.ndarray] = None

        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            replay_buffer_class=replay_buffer_class or MaskedReplayBuffer,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            max_grad_norm=max_grad_norm,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

    def _setup_model(self) -> None:
        # The checks live here and not in __init__: on `load()` without an env
        # the spaces are not known yet when the constructor runs.
        if not isinstance(self.action_space, spaces.Discrete):
            raise ValueError("MaskedCDQN only supports a discrete action space.")
        super()._setup_model()
        # `load()` takes the policy class from the artifact itself, so a model
        # saved by a plain stable_baselines3.DQN would otherwise come up as a
        # MaskedCDQN with an unmasked Q-network and fail on the first
        # q_net(obs, mask) call.
        if not isinstance(self.policy, MaskedDQNPolicy):
            raise ValueError(
                f"MaskedCDQN requires a masked policy, got {type(self.policy).__name__}. "
                "An artifact saved by a plain stable_baselines3.DQN must be loaded with DQN."
            )

    # ------------------------------------------------------------------
    # Action masks
    # ------------------------------------------------------------------

    @property
    def n_actions(self) -> int:
        return int(self.action_space.n)

    def query_action_masks(self, env=None) -> np.ndarray:
        """Current masks of valid actions of all sub-environments."""

        env = env if env is not None else self.env
        masks = env.env_method("get_action_mask")
        return np.asarray(np.stack(masks), dtype=np.bool_)

    def _random_valid_action(self, mask: np.ndarray) -> int:
        valid_actions = np.flatnonzero(np.asarray(mask, dtype=np.bool_))
        if valid_actions.size == 0:
            raise RuntimeError(
                "No valid actions. The environment must end the episode before an action is chosen."
            )
        return int(np.random.choice(valid_actions))

    def _resolve_action_masks(self, action_masks, batch_size: int) -> np.ndarray:
        if action_masks is None:
            action_masks = self._last_action_masks
        if action_masks is None:
            return np.ones((batch_size, self.n_actions), dtype=np.bool_)
        masks = np.asarray(action_masks, dtype=np.bool_)
        if masks.ndim == 1:
            masks = masks[None, ...]
        if masks.shape != (batch_size, self.n_actions):
            raise ValueError(
                f"action_masks has shape {masks.shape}, expected ({batch_size}, {self.n_actions})."
            )
        return masks

    # ------------------------------------------------------------------
    # Data collection loop
    # ------------------------------------------------------------------

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ):
        result = super()._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )
        # super() has already reset the environment, so the masks match _last_obs.
        self._last_action_masks = self.query_action_masks()
        return result

    def _sample_action(
        self,
        learning_starts: int,
        action_noise=None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        masks = self._resolve_action_masks(None, n_envs)
        if self.num_timesteps < learning_starts or np.random.rand() < self.exploration_rate:
            action = np.array([self._random_valid_action(mask) for mask in masks], dtype=np.int64)
        else:
            action, _ = self.predict(self._last_obs, deterministic=True, action_masks=masks)
            action = np.asarray(action, dtype=np.int64).reshape(n_envs)
        return action, action

    def _store_transition(
        self,
        replay_buffer,
        buffer_action: np.ndarray,
        new_obs: np.ndarray,
        reward: np.ndarray,
        dones: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        # The environment has already stepped and finished sub-environments have
        # auto-reset, so the mask after the step belongs to the new episode. For
        # a finished transition the next mask is empty: there is nothing to
        # bootstrap from.
        env_masks = self.query_action_masks()
        dones_flat = np.asarray(dones).reshape(-1).astype(bool)
        next_masks = np.where(dones_flat[:, None], False, env_masks)

        current_masks = self._resolve_action_masks(None, len(infos))
        infos = [dict(info) for info in infos]
        for index, info in enumerate(infos):
            info["action_mask"] = current_masks[index]
            info["next_action_mask"] = next_masks[index]

        super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)
        self._last_action_masks = env_masks

    def predict(
        self,
        observation: np.ndarray,
        state=None,
        episode_start=None,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, Optional[tuple]]:
        observation = np.asarray(observation)
        is_vectorized = observation.ndim == len(self.observation_space.shape) + 1
        batch_size = observation.shape[0] if is_vectorized else 1
        masks = self._resolve_action_masks(action_masks, batch_size)

        if not deterministic and np.random.rand() < self.exploration_rate:
            action = np.array([self._random_valid_action(mask) for mask in masks], dtype=np.int64)
        else:
            self.policy.set_training_mode(False)
            obs_tensor, _ = self.policy.obs_to_tensor(observation)
            mask_tensor = th.as_tensor(masks, dtype=th.bool, device=self.device)
            with th.no_grad():
                q_values = self.q_net(obs_tensor, mask_tensor)
                q_values = q_values.masked_fill(~mask_tensor, -th.inf)
            action = q_values.argmax(dim=1).cpu().numpy().astype(np.int64)

        if not is_vectorized:
            action = action[0]
        return action, state

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """Gradient steps on the masked (C-)DQN loss.

        With online network ``Q``, target network ``Q'``, masks ``M`` and
        Huber loss ``l``::

            a*     = argmax_{a in M(s')} Q(s', a)            (no gradient)
            y_DQN  = r + gamma * (1 - done) * Q'(s', a*)      (no gradient)
            y_MSBE = r + gamma * (1 - done) * max_{a in M(s')} Q(s', a)
            L      = mean(max(l(Q(s, a), y_DQN), l(Q(s, a), y_MSBE)))

        The gradient flows through ``y_MSBE`` (residual gradient). States with no
        valid next action bootstrap from zero. With ``use_cdqn=False`` the loss
        is ``mean(l(Q(s, a), y_DQN))``.
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses: list[float] = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            masks = replay_data.action_masks
            next_masks = replay_data.next_action_masks

            if not masks.gather(1, replay_data.actions.long()).all():
                raise RuntimeError("The replay buffer contains an action that is invalid under its mask.")

            current_q = self.q_net(replay_data.observations, masks).gather(
                1, replay_data.actions.long()
            )

            # A single differentiable forward pass of the online network over
            # next_observations. The DQN branch detaches it for action selection
            # (Double DQN), the MSBE branch keeps the gradient. Reusing it keeps
            # the number of forward passes the same as in plain DQN.
            next_q_online = self.q_net(replay_data.next_observations, next_masks)
            has_valid_next = next_masks.any(dim=1, keepdim=True)

            with th.no_grad():
                next_q_selection = next_q_online.detach().masked_fill(~next_masks, -th.inf)
                next_actions = next_q_selection.argmax(dim=1, keepdim=True)

                next_q_target = self.q_net_target(replay_data.next_observations, next_masks).gather(
                    1, next_actions
                )
                next_q_target = th.where(has_valid_next, next_q_target, th.zeros_like(next_q_target))

                target_dqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_target

            loss_dqn = F.smooth_l1_loss(current_q, target_dqn, reduction="none")

            if self.use_cdqn:
                # l_MSBE: the bootstrap goes through the online network and its
                # gradient is NOT disabled; this is the residual-gradient term of
                # Eq. 12 (Wang & Ueda). -inf at masked positions is safe: max
                # picks a valid position and torch.where drops degenerate rows
                # before any arithmetic, so neither forward nor backward yields NaN.
                next_q_bootstrap = next_q_online.masked_fill(~next_masks, -th.inf)
                next_q_bootstrap = next_q_bootstrap.max(dim=1, keepdim=True).values
                next_q_bootstrap = th.where(
                    has_valid_next,
                    next_q_bootstrap,
                    th.zeros_like(next_q_bootstrap),
                )

                target_msbe = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_bootstrap
                loss_msbe = F.smooth_l1_loss(current_q, target_msbe, reduction="none")

                # Element-wise max, then the mean: L_CDQN = E[max(l_DQN, l_MSBE)].
                loss = th.max(loss_dqn, loss_msbe).mean()
                msbe_mean = float(loss_msbe.mean().item())
                # Share of the batch where the MSBE branch is active. It grows as
                # training approaches divergence, so it is an early warning signal.
                msbe_active = float((loss_msbe > loss_dqn).float().mean().item())
            else:
                loss = loss_dqn.mean()
                msbe_mean = float("nan")
                msbe_active = float("nan")

            loss_value = float(loss.item())
            if not np.isfinite(loss_value):
                raise RuntimeError(f"Non-finite loss at step {self.num_timesteps}: {loss_value}")
            losses.append(loss_value)

            self.policy.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            # Divergence diagnostics: without them growth of |Q| is only visible
            # after the loss has exploded.
            self.last_update_info = {
                "loss": loss_value,
                "loss_dqn": float(loss_dqn.mean().item()),
                "loss_msbe": msbe_mean,
                "msbe_active_fraction": msbe_active,
                "q_mean": float(current_q.mean().item()),
                "q_abs_max": float(current_q.abs().max().item()),
                "target_mean": float(target_dqn.mean().item()),
                "grad_norm": float(grad_norm),
            }

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", float(np.mean(losses)) if losses else float("nan"))
        self.logger.record("train/use_cdqn", int(self.use_cdqn))
        for key, value in self.last_update_info.items():
            if np.isfinite(value):
                self.logger.record(f"train/{key}", value)

    # ------------------------------------------------------------------
    # Warm start: replay prefill + behavioural cloning
    # ------------------------------------------------------------------

    def hot_start(
        self,
        *,
        greedy_order: list[str],
        bc_config: Optional[BehavioralCloningConfig] = None,
        prefill_episodes: int = 3,
        max_steps: Optional[int] = None,
        output_dir: Optional[Path] = None,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Warm the agent up with the greedy policy before Q-learning.

        First the replay buffer is filled with noise-free greedy episodes; then,
        if BC is enabled, extra noisy demonstrations are collected and the
        Q-network is trained with supervision to repeat the greedy action.

        Call before ``learn()``: ``learn()`` resets the environment itself and
        starts from the warmed-up buffer and weights.
        """

        bc_config = bc_config if bc_config is not None else BehavioralCloningConfig()
        rng = np.random.default_rng(int(seed) + 10_007)
        max_steps = int(max_steps) if max_steps is not None else max(1, self.n_actions * 2)

        prefill_transitions: list[dict[str, Any]] = []
        prefill_stats: dict[str, Any] = {"episodes": 0, "transitions": 0}
        if prefill_episodes > 0:
            prefill_transitions, prefill_stats = collect_greedy_demonstrations(
                env=self.env,
                greedy_order=greedy_order,
                n_actions=self.n_actions,
                episodes=int(prefill_episodes),
                max_steps=max_steps,
                noise_prob=0.0,
                rng=rng,
            )
            store_demonstrations(self.replay_buffer, prefill_transitions)
            print(
                "Greedy replay prefill: "
                f"episodes={prefill_stats['episodes']} | "
                f"transitions={prefill_stats['transitions']} | "
                f"fallback_actions={prefill_stats['fallback_actions']} | "
                f"plan_sizes={prefill_stats['plan_sizes']} | "
                f"final_npvs={prefill_stats['final_npvs']} | "
                f"replay_size={self.replay_buffer.size()}"
            )
        prefill_stats["enabled"] = bool(prefill_episodes > 0)

        bc_transitions: list[dict[str, Any]] = []
        bc_demo_stats: dict[str, Any] = {}
        if bc_config.enabled:
            if bc_config.reuse_prefill_demos:
                bc_transitions.extend(prefill_transitions)

            if bc_config.demo_episodes > 0:
                extra, bc_demo_stats = collect_greedy_demonstrations(
                    env=self.env,
                    greedy_order=greedy_order,
                    n_actions=self.n_actions,
                    episodes=int(bc_config.demo_episodes),
                    max_steps=max_steps,
                    noise_prob=bc_config.demo_noise_prob,
                    rng=rng,
                )
                if bc_config.add_demos_to_replay:
                    store_demonstrations(self.replay_buffer, extra)
                bc_transitions.extend(extra)
                print(
                    "Behavioral cloning demonstrations: "
                    f"episodes={bc_demo_stats['episodes']} | "
                    f"transitions={bc_demo_stats['transitions']} | "
                    f"noise_actions={bc_demo_stats['noise_actions']} | "
                    f"fallback_actions={bc_demo_stats['fallback_actions']} | "
                    f"added_to_replay={bool(bc_config.add_demos_to_replay)} | "
                    f"replay_size={self.replay_buffer.size()}"
                )

        normalize_obs = getattr(self._vec_normalize_env, "normalize_obs", None)
        bc_stats = pretrain_behavioral_cloning(
            model=self,
            transitions=bc_transitions,
            cfg=bc_config,
            normalize_obs=normalize_obs,
            output_dir=output_dir,
            rng=rng,
        )
        bc_stats["demonstrations"] = bc_demo_stats
        bc_stats["reused_prefill_transitions"] = (
            int(len(prefill_transitions)) if bc_config.enabled and bc_config.reuse_prefill_demos else 0
        )

        if bc_stats.get("trained"):
            self._apply_post_bc_exploration(bc_config)

        return {"greedy_prefill": prefill_stats, "behavioral_cloning": bc_stats}

    def _apply_post_bc_exploration(self, bc_config: BehavioralCloningConfig) -> None:
        """Lower the initial epsilon so that the cloned policy is actually used."""

        new_eps = bc_config.post_bc_exploration_initial_eps
        if new_eps is None:
            if self.exploration_initial_eps > 0.5:
                print(
                    "WARNING: cloning trained the policy, but "
                    f"exploration_initial_eps={self.exploration_initial_eps} keeps actions almost "
                    "random for the whole decay window. Set post_bc_exploration_initial_eps."
                )
            return

        print(
            "Behavioral cloning pretrained the policy: "
            f"exploration_initial_eps {self.exploration_initial_eps} -> {float(new_eps)}"
        )
        self.exploration_initial_eps = float(new_eps)
        self.exploration_schedule = _linear_schedule(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )
        self.exploration_rate = self.exploration_initial_eps

    # ------------------------------------------------------------------
    # Deterministic evaluation
    # ------------------------------------------------------------------

    def run_deterministic_episode(self, env=None, max_steps: Optional[int] = None) -> dict[str, Any]:
        """One episode with epsilon=0. VecNormalize statistics are frozen meanwhile."""

        env = env if env is not None else self.env
        max_steps = int(max_steps) if max_steps is not None else max(1, self.n_actions * 2)

        previous_training = getattr(env, "training", None)
        previous_norm_reward = getattr(env, "norm_reward", None)
        if previous_training is not None:
            env.training = False
        if previous_norm_reward is not None:
            env.norm_reward = False

        try:
            obs = env.reset()
            masks = self.query_action_masks(env)
            for _ in range(max_steps):
                action, _ = self.predict(obs, deterministic=True, action_masks=masks)
                obs, _, dones, infos = env.step(np.asarray(action, dtype=np.int64).reshape(-1))
                info = dict(infos[0])
                if bool(dones[0]):
                    plan = info.get("final_plan")
                    return {
                        "final_npv": float(info.get("final_npv", np.nan)),
                        "plan_size": len(plan.well_plans) if plan is not None else 0,
                        "order": (
                            [str(context.well.name) for context in plan.well_plans]
                            if plan is not None
                            else []
                        ),
                    }
                masks = self.query_action_masks(env)
        finally:
            if previous_training is not None:
                env.training = previous_training
            if previous_norm_reward is not None:
                env.norm_reward = previous_norm_reward

        raise RuntimeError("Deterministic evaluation did not finish the episode.")
