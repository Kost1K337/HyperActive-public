"""SB3 Q-network and policy for observations of shape ``[batch, n_actions, obs_dim]``.

Every row of a :class:`~hyperactive.env.PlanEnv` observation describes one
candidate action, and invalid actions are padded with zero rows. The standard
stable-baselines3 ``MlpPolicy`` flattens such an observation and loses the
"row = action" structure. Here every row is encoded by a shared encoder and the
Q-value of an action is computed from its row embedding and a context vector
obtained by averaging the embeddings of the valid rows::

    h_i = f(LayerNorm(x_i))
    c   = sum_i m_i h_i / max(1, sum_i m_i)
    Q_i = g([h_i, c])

The network is equivariant to permutations of the candidate rows and its
parameters do not depend on the number of actions.

Masking is not applied inside ``forward``: the mask is only used for the
context. Invalid actions are excluded by action selection and target
computation.
"""

from __future__ import annotations

from typing import Optional

import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from torch import nn

INVALID_ACTION_FILL = -1.0e9
"""Finite replacement for ``-inf`` when masking Q-values inside a loss.

Action selection and Q-learning targets may safely use ``-inf``: they take an
argmax over a row that contains at least one valid action. Behavioural cloning
passes masked Q-values through ``cross_entropy`` and ``max(Q + margin)``, where
``-inf`` turns into NaN gradients, so a large finite value is used there.
"""


class MaskedQNetwork(QNetwork):
    """Q-network taking an observation ``[batch, n_actions, obs_dim]`` and an action mask.

    The constructor signature matches ``stable_baselines3.dqn.policies.QNetwork``,
    so the network plugs into ``DQNPolicy.make_q_net`` unchanged. ``net_arch``
    sets the hidden width; only its first element is used.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch: Optional[list[int]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ) -> None:
        # QNetwork.__init__ builds its own MLP head from net_arch; it is not
        # used and is replaced right below.
        super().__init__(
            observation_space,
            action_space,
            features_extractor,
            features_dim,
            net_arch=net_arch,
            activation_fn=activation_fn,
            normalize_images=normalize_images,
        )

        if len(observation_space.shape) != 2:
            raise ValueError(
                "MaskedQNetwork expects an observation_space of shape (n_actions, obs_dim), "
                f"got {observation_space.shape}."
            )

        self.n_actions = int(action_space.n)
        self.obs_dim = int(observation_space.shape[1])
        if observation_space.shape[0] != self.n_actions:
            raise ValueError(
                f"The number of observation rows {observation_space.shape[0]} does not match "
                f"the number of actions {self.n_actions}."
            )

        hidden_dim = int(net_arch[0]) if net_arch else 256
        self.hidden_dim = hidden_dim

        # Replace the QNetwork MLP head with the "row encoder + context" architecture.
        self.q_net = nn.Identity()
        self.row_encoder = nn.Sequential(
            nn.LayerNorm(self.obs_dim),
            nn.Linear(self.obs_dim, hidden_dim),
            activation_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn(),
        )
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            activation_fn(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: th.Tensor, action_mask: Optional[th.Tensor] = None) -> th.Tensor:
        obs = obs.float()
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
        if obs.dim() != 3:
            raise ValueError(f"obs must have shape [batch, n_actions, obs_dim], got {tuple(obs.shape)}")
        if obs.shape[1] != self.n_actions or obs.shape[2] != self.obs_dim:
            raise ValueError(
                f"Expected shape [batch, {self.n_actions}, {self.obs_dim}], got {tuple(obs.shape)}"
            )

        batch_size = obs.shape[0]
        if action_mask is None:
            action_mask = th.ones((batch_size, self.n_actions), dtype=th.bool, device=obs.device)
        else:
            action_mask = action_mask.to(device=obs.device, dtype=th.bool)
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            if action_mask.shape != obs.shape[:2]:
                raise ValueError(
                    f"action_mask must have shape {tuple(obs.shape[:2])}, got {tuple(action_mask.shape)}"
                )

        encoded = self.row_encoder(obs)

        valid = action_mask.float().unsqueeze(-1)
        denominator = valid.sum(dim=1).clamp_min(1.0)
        context = (encoded * valid).sum(dim=1) / denominator
        context = context.unsqueeze(1).expand(-1, self.n_actions, -1)

        return self.q_head(th.cat([encoded, context], dim=-1)).squeeze(-1)

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self(observation).argmax(dim=1).reshape(-1)


class MaskedDQNPolicy(DQNPolicy):
    """``DQNPolicy`` with :class:`MaskedQNetwork` instead of the standard Q-network."""

    def make_q_net(self) -> MaskedQNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return MaskedQNetwork(**net_args).to(self.device)


class MaskedMlpPolicy(MaskedDQNPolicy):
    """Default alias so that ``policy="MlpPolicy"`` works with ``MaskedCDQN``."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        **kwargs,
    ) -> None:
        kwargs.setdefault("features_extractor_class", FlattenExtractor)
        kwargs.setdefault("normalize_images", False)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
