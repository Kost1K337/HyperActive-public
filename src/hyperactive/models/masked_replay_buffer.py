"""SB3 replay buffer that additionally stores masks of valid actions.

A mask is needed both for the state and for the next state: the first one
checks that the stored action was valid, the second one is used to build the
Double-DQN and C-DQN targets. The standard ``ReplayBuffer`` knows nothing about
masks, so they are passed through ``infos`` in the same way SB3 passes
``terminal_observation``.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize


class MaskedReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    action_masks: th.Tensor
    next_action_masks: th.Tensor


class MaskedReplayBuffer(ReplayBuffer):
    """``ReplayBuffer`` with ``action_mask`` and ``next_action_mask`` fields.

    ``add`` reads both masks from ``infos[i]``. If they are missing, the action
    is treated as valid and the next state as fully valid, or as terminal for
    finished transitions; the behaviour then reduces to plain DQN.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ) -> None:
        if optimize_memory_usage:
            raise ValueError("MaskedReplayBuffer does not support optimize_memory_usage.")
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=False,
            handle_timeout_termination=handle_timeout_termination,
        )
        self.n_actions = int(action_space.n)
        self.action_masks = np.ones((self.buffer_size, self.n_envs, self.n_actions), dtype=np.bool_)
        self.next_action_masks = np.ones((self.buffer_size, self.n_envs, self.n_actions), dtype=np.bool_)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        position = self.pos
        super().add(obs, next_obs, action, reward, done, infos)

        done = np.asarray(done).reshape(self.n_envs)
        for env_index, info in enumerate(infos):
            mask = info.get("action_mask")
            next_mask = info.get("next_action_mask")
            if mask is None:
                mask = np.ones(self.n_actions, dtype=np.bool_)
            if next_mask is None:
                next_mask = (
                    np.zeros(self.n_actions, dtype=np.bool_)
                    if bool(done[env_index])
                    else np.ones(self.n_actions, dtype=np.bool_)
                )
            self.action_masks[position, env_index] = np.asarray(mask, dtype=np.bool_)
            self.next_action_masks[position, env_index] = np.asarray(next_mask, dtype=np.bool_)

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> MaskedReplayBufferSamples:
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        observations = self._normalize_obs(self.observations[batch_inds, env_indices, :], env)
        next_observations = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        return MaskedReplayBufferSamples(
            observations=self.to_torch(observations),
            actions=self.to_torch(self.actions[batch_inds, env_indices, :]),
            next_observations=self.to_torch(next_observations),
            dones=self.to_torch(
                (
                    self.dones[batch_inds, env_indices]
                    * (1 - self.timeouts[batch_inds, env_indices])
                ).reshape(-1, 1)
            ),
            rewards=self.to_torch(
                self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)
            ),
            action_masks=th.as_tensor(
                self.action_masks[batch_inds, env_indices, :], device=self.device, dtype=th.bool
            ),
            next_action_masks=th.as_tensor(
                self.next_action_masks[batch_inds, env_indices, :], device=self.device, dtype=th.bool
            ),
        )
