"""Masked C-DQN on stable-baselines3.

:class:`MaskedCDQN` is the only neural network of the project: DQN from
stable-baselines3 with masking of invalid actions, an optional warm start by
behavioural cloning of the greedy policy, and the C-DQN loss instead of the
plain DQN loss.
"""

from hyperactive.models.behavioral_cloning import (
    BehavioralCloningConfig,
    behavioral_cloning_loss,
    collect_greedy_demonstrations,
    pretrain_behavioral_cloning,
    store_demonstrations,
)
from hyperactive.models.masked_cdqn import MaskedCDQN
from hyperactive.models.masked_q_network import (
    INVALID_ACTION_FILL,
    MaskedDQNPolicy,
    MaskedMlpPolicy,
    MaskedQNetwork,
)
from hyperactive.models.masked_replay_buffer import MaskedReplayBuffer, MaskedReplayBufferSamples

__all__ = [
    "INVALID_ACTION_FILL",
    "BehavioralCloningConfig",
    "MaskedCDQN",
    "MaskedDQNPolicy",
    "MaskedMlpPolicy",
    "MaskedQNetwork",
    "MaskedReplayBuffer",
    "MaskedReplayBufferSamples",
    "behavioral_cloning_loss",
    "collect_greedy_demonstrations",
    "pretrain_behavioral_cloning",
    "store_demonstrations",
]
