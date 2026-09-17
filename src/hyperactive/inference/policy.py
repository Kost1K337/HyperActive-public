"""Loading a trained policy and building plans with it."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hyperactive.core import Plan, TeamPool
from hyperactive.env import FEATURE_SET, PlanEnv
from hyperactive.models import MaskedCDQN
from hyperactive.planning import LinearProductionProfile, SimpleTeamMovement

MANIFEST_NAME = "manifest.json"


class ModelManifestError(RuntimeError):
    """The model files do not match their manifest."""


@dataclass
class PolicyBundle:
    """A trained Q-network together with its observation normalisation statistics."""

    model: MaskedCDQN
    vec_normalize: VecNormalize
    manifest: dict[str, Any]
    directory: Path

    @property
    def n_actions(self) -> int:
        return int(self.manifest["action_window"])


def sha256_of(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _NoopCost:
    def compute(self, context):
        return context


def _shape_env(n_actions: int) -> DummyVecEnv:
    """An environment without wells, used only for the spaces and VecNormalize loading."""
    return DummyVecEnv(
        [
            lambda: PlanEnv(
                wells=[],
                team_pool=TeamPool(),
                movement=SimpleTeamMovement(),
                production_profile=LinearProductionProfile(),
                cost_function=_NoopCost(),
                n_actions=n_actions,
            )
        ]
    )


def load_policy(model_dir: str | Path, device: str = "cpu") -> PolicyBundle:
    """Load ``model.zip`` and ``vec_normalize.pkl`` described by ``manifest.json``.

    The manifest pins the sha256 of both files and the feature set the network
    was trained on. Feature sets of equal width are indistinguishable by shape,
    so a mismatch would silently feed features into the wrong input positions;
    it is therefore checked explicitly.
    """
    directory = Path(model_dir)
    manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))

    if manifest.get("feature_set") != FEATURE_SET:
        raise ModelManifestError(
            f"Model was trained on feature set {manifest.get('feature_set')!r}, "
            f"this code produces {FEATURE_SET!r}."
        )
    files = manifest["files"]
    for role in ("model", "vec_normalize"):
        path = directory / files[role]["path"]
        actual = sha256_of(path)
        if actual != files[role]["sha256"]:
            raise ModelManifestError(
                f"{path.name}: sha256 {actual[:12]} does not match the manifest "
                f"({files[role]['sha256'][:12]})."
            )

    shape_env = _shape_env(int(manifest["action_window"]))
    model = MaskedCDQN.load(
        str(directory / files["model"]["path"]),
        custom_objects={
            "observation_space": shape_env.observation_space,
            "action_space": shape_env.action_space,
        },
        device=device,
    )
    vec_normalize = VecNormalize.load(str(directory / files["vec_normalize"]["path"]), shape_env)
    vec_normalize.training = False
    vec_normalize.norm_reward = False
    return PolicyBundle(model=model, vec_normalize=vec_normalize, manifest=manifest, directory=directory)


def _runtime_env(env: PlanEnv, bundle: PolicyBundle) -> VecNormalize:
    runtime = VecNormalize(DummyVecEnv([lambda: env]), training=False, norm_obs=True, norm_reward=True)
    runtime.obs_rms = bundle.vec_normalize.obs_rms
    runtime.ret_rms = bundle.vec_normalize.ret_rms
    runtime.training = False
    runtime.norm_reward = False
    return runtime


def run_episode(
    env: PlanEnv,
    bundle: PolicyBundle,
    exploration: float = 0.0,
    seed: int = 0,
    max_steps: int | None = None,
) -> tuple[Plan, dict[str, Any]]:
    """Build one plan.

    With ``exploration=0`` every step takes the valid action with the highest
    Q-value. With ``exploration=p`` a random valid action is taken with
    probability ``p`` (seeded by ``seed``).
    """
    if env.action_space.n != bundle.n_actions:
        raise ValueError(
            f"Environment has {env.action_space.n} actions, the policy expects {bundle.n_actions}."
        )
    runtime = _runtime_env(env, bundle)
    obs = runtime.reset()
    rng = np.random.default_rng(seed)
    steps = max_steps if max_steps is not None else len(env.wells) + 1
    for _ in range(max(steps, 1)):
        action = None
        if exploration > 0.0 and float(rng.random()) < exploration:
            valid = np.flatnonzero(env.get_action_mask())
            if valid.size:
                action = np.array([int(rng.choice(valid))], dtype=np.int64)
        if action is None:
            masks = bundle.model.query_action_masks(runtime)
            action = bundle.model.predict(obs, deterministic=True, action_masks=masks)[0]
        obs, _, dones, infos = runtime.step(action)
        info = infos[0] if infos else {}
        if bool(dones[0]):
            return info.get("final_plan", env.plan), info
    raise RuntimeError("The episode did not finish within the expected number of steps.")


def plan_with_policy(
    env: PlanEnv,
    bundle: PolicyBundle,
    episodes: int = 1,
    exploration: float = 0.0,
) -> tuple[Plan, dict[str, Any]]:
    """Best of ``episodes`` plans by NPV; episode ``k`` is seeded with ``k``.

    ``PlanEnv.reset`` recreates the plan, the crew manager and a copy of the
    crew pool, so the same environment is reused across episodes. With
    ``episodes=1`` and ``exploration=0`` this is a single deterministic rollout.
    """
    results = []
    for episode in range(max(1, int(episodes))):
        plan, info = run_episode(env, bundle, exploration=exploration, seed=episode)
        npv = float(info.get("final_npv", plan.total_profit()))
        results.append((npv, len(plan.well_plans), episode, plan))
    npv, size, episode, plan = max(results, key=lambda item: (item[0], item[1]))
    stats = {
        "episodes": len(results),
        "exploration": exploration,
        "selected_episode": episode,
        "npvs": [item[0] for item in results],
    }
    return plan, stats
