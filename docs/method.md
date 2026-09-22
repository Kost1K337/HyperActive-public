# Method

This document describes the planning problem, the Markov decision process
(MDP) built on top of it, the feature encoding, and the learning algorithm
(masked C-DQN with an optional behavioural-cloning warm start). It is meant
for readers checking the math and the RL design, not for API usage — see
[`reproducing.md`](reproducing.md) for that.

## 1. The planning problem

A **plan** is an ordered sequence of wells assigned to drilling and GTM
(well-intervention) crews. Each well has a task chain (e.g. drilling followed
by hydraulic fracturing); each task is assigned to a crew, which schedules it
on its calendar taking travel time between clusters into account
(`hyperactive.planning.team_manager.TeamManager`).

Two horizons bound a plan:

* the **economic horizon** `end` — the point up to which NPV is accumulated;
* the **work window** `end_jobs ≤ end` — the deadline by which every task must
  finish. A well whose schedule would run past `end_jobs` is infeasible and is
  dropped from consideration.

At every step of plan construction, every remaining well is scheduled
tentatively (same procedure for the greedy planner and the RL environment):
crew assignment, production profile, and NPV are computed for each candidate;
candidates that violate a constraint or finish after `end_jobs` are removed.

### NPV (`hyperactive.planning.cost.NPV`)

For a well with launch date `t0` (end of its last task), project start `T0`,
`s = (t0 - T0).days / 365` and discount rate `r`:

```
NPV = Σ_m (P · oil_m − opex_m) / (1 + r)^(s + m/12)
      − capex / (1 + r)^s
      − travel_days · travel_cost_per_day
```

`oil_m` and `opex_m` are the monthly oil volume and operating cost from the
well's production profile; `capex` is `cost_per_metre[well_type] · length +
equipment_cost`; the travel term (crew move preceding drilling) is **not**
discounted. Plan NPV is the sum of well NPVs (`Plan.total_profit()`).

### Production profile

`ArpsDeclineProductionProfile` (the profile used for both training and the
released model) gives a hyperbolic decline:

```
q(t) = q0 / (1 + b·D·t)^(1/b)
```

with `t` in years since launch, `D = 0.175`, `b = 1.548` by default. A
`LinearProductionProfile` (constant rate) is also provided for baselines.

### Greedy baseline (`hyperactive.greedy.PlanBuilder`)

At every step, among the feasible candidates, take the one with the largest
`NPV − drill_team_penalty` (a small penalty discouraging unnecessary parallel
crews on one cluster). This is *not* "the cheapest well first" — `.cost` is
discounted NPV, so the greedy rule maximises expected value added per step,
not minimises spend.

## 2. The MDP (`hyperactive.env.PlanEnv`)

| | |
|---|---|
| **State** | The set of feasible candidates at the current step (built by the same procedure as the greedy planner), sorted by NPV and truncated to `n_actions` (8 for the released model). |
| **Observation** | A `float32` matrix `[n_actions, 39]`: one 39-dimensional feature row per candidate (§3), zero-padded past the number of real candidates. |
| **Action** | An index into the candidate list. Valid actions are the first `len(candidates)` indices; the boolean mask is exposed via `info["action_mask"]` / `PlanEnv.get_action_mask()`, because a padded row is no longer exactly zero after `VecNormalize` normalises the observation. |
| **Reward** | The increase in plan NPV caused by the step, so the undiscounted episode return equals the final plan's NPV. Choosing a padded (invalid) action ends the episode with reward `-10000`. |
| **Termination** | No feasible candidates remain (pool exhausted, work window elapsed, or every remaining well violates a constraint). |

Because the network is trained to pick the argmax of masked Q-values, the
policy always emits an action-space index; `PlanEnv` translates it to "the
candidate at that rank" — the mapping from index to well changes every step.

## 3. Feature encoding (`hyperactive.env.features`, feature set `gen39`)

Each of the 39 features is defined relative to the *planning front* (the
earliest start among current candidates) and, where relevant, normalised by a
quantity that does not grow with pool size — so a policy trained on 16-well
subsets transfers to pools of hundreds of wells without leaving the training
distribution. See the module docstring for the full design rationale; the
blocks are:

| Block | Width | Features |
|---|---:|---|
| **Base** | 20 | `oil_rate`, `liq_rate`, `length` — well physics. `cost_npv`, `cash_flow`, `capex`, `travel_cost` — candidate economics. `ready_in_days`, `has_ready_date`, `job1_start_days`, `job1_duration_days`, `job1_travel_days`, `has_job2`, `job2_gap_days`, `job2_duration_days`, `job2_travel_days`, `finish_in_days` — schedule, in days from the planning front. `jobs_window_left_days`, `placed_share`, `same_cluster` — plan state. |
| **Cluster (pad)** | 3 | `pad_wells_left_current`, `pad_wells_left_candidate`, `pad_closes` — how many wells remain on the current/candidate cluster, and whether taking the candidate closes its cluster. Normalised by total pool size. |
| **Value** | 2 | `value_density_norm` (NPV per crew-day, min–max normalised within the current candidate window), `value_rank` (share of window candidates with lower density) — a network cannot compute ranks itself since row encoding is permutation-averaged, so rank is precomputed. |
| **Dynamic crew capacity** | 5 | `drill_capacity_left`, `gtm_capacity_left` — free crew-days remaining in the work window. `crew_bottleneck` — which crew type binds. `oil_window_left`, `oil_window_after` — headroom under an annual oil cap constraint (1.0 when no cap is active), before and after taking the candidate. |
| **Production profile** | 2 | `horizon_coverage` — share of the economic horizon the well actually produces for. `decline_year` — ratio of average daily rate a year in to the first full month, capturing decline speed independent of the profile model used. |
| **Arrival economics** | 3 | `best_arriving_value_ratio`, `wait_idle_share`, `arrivals_better_share` — whether it pays to wait for a better well to become ready rather than drilling the candidate now, and what waiting costs in idle crew-days. |
| **Configuration** | 4 | `jobs_window_years`, `econ_window_ratio` (economic horizon / work window, capped at 4), `job_window_share`, `finish_window_share` — scenario-level context that does not depend on pool size. |

`hyperactive.env.FEATURE_SET` (`"gen39"`) is stored in every model manifest
and checked when a policy is loaded, so a feature-encoding mismatch between a
saved model and the running code fails loudly instead of silently feeding
inputs into the wrong positions.

## 4. Q-network (`hyperactive.models.masked_q_network.MaskedQNetwork`)

The observation has structure — "row = candidate action" — that a standard
flattening MLP policy would discard. Instead, every row is encoded by a
shared encoder, and Q-values are computed from the row embedding together
with a context vector averaged over the *valid* rows only:

```
h_i = f(LayerNorm(x_i))
c   = Σ_i m_i h_i / max(1, Σ_i m_i)
Q_i = g([h_i, c])
```

where `m_i` is the action mask. This makes the network equivariant to
permutations of the candidate rows, and its parameter count does not depend
on the number of actions — a policy trained with an 8-wide action window can
in principle be fine-tuned onto a wider one by reusing the row encoder.
Masking is applied to the *context*, not inside `forward`; invalid actions
are excluded downstream, at action selection and target computation.

## 5. Learning algorithm (`hyperactive.models.masked_cdqn.MaskedCDQN`)

`MaskedCDQN` extends `stable_baselines3.DQN` with three changes:

**Action masking.** Invalid actions are excluded from ε-greedy exploration,
from the online-network argmax used for action selection, from the
Double-DQN target computation, and from behavioural-cloning losses. The mask
is stored in the replay buffer alongside each transition
(`MaskedReplayBuffer`).

**C-DQN loss** (Wang & Ueda, ICLR 2022). With online network `Q`, target
network `Q'`, and Huber loss `l`:

```
a*     = argmax_{a in mask(s')} Q(s', a)              (no gradient)
y_DQN  = r + γ·(1 − done)·Q'(s', a*)                   (no gradient, Double DQN)
y_MSBE = r + γ·(1 − done)·max_{a in mask(s')} Q(s', a)  (gradient flows through Q)
L      = E[ max( l(Q(s,a), y_DQN), l(Q(s,a), y_MSBE) ) ]
```

`y_MSBE` bootstraps through the *online* network with its gradient enabled
(a residual-gradient term), which upper-bounds the next-step DQN loss and
makes the sequence of loss minima non-increasing — this is what prevents Q
divergence relative to plain DQN. Both branches share one forward pass over
`next_observations`, so C-DQN costs no extra inference compared to plain
masked Double DQN. With `use_cdqn=False` the loss reduces to
`E[l(Q(s,a), y_DQN)]` (plain masked Double DQN); this is the ablation control
used to measure C-DQN's contribution.

**Behavioural-cloning warm start** (`hyperactive.models.behavioral_cloning`,
optional, **on** for the released `bc39` model — the stage the model is named
after). Before Q-learning begins, the replay buffer can be pre-filled with
noise-free greedy-policy episodes, and the Q-network can be pretrained with
supervision to reproduce the greedy action:

* `loss_type="cross_entropy"` treats masked Q-values as softmax-policy logits
  (standard BC).
* `loss_type="margin"` is the DQfD large-margin loss: only requires the
  demonstrated action to lead the runner-up by a fixed margin, which distorts
  the Q scale less going into Q-learning.

A configurable fraction of demonstration steps takes a random valid action
instead of the greedy one while keeping the greedy label — this shows the
cloned policy states off the greedy trajectory, so it survives its own
mistakes instead of collapsing after the first deviation. After cloning, the
exploration schedule's initial ε is lowered (`post_bc_exploration_initial_eps`)
so the pretrained policy is actually used rather than overridden by near-random
exploration for the whole decay window. `bc39` was cloned on 30 subsets with
the cross-entropy loss and ε then started at 0.3 instead of 1.0; the greedy
replay prefill was left off, so cloning is the only thing the policy inherits
from the baseline it is measured against.

## 6. Training regime (`experiments/train.py`)

Wells are drawn as *random subsets* of a larger pool on every episode reset
(`RandomSubsetPlanEnv`), with the pool split up-front into a training and a
held-out part — training on a fixed subset would let the policy memorise a
well order for that specific set rather than learn a transferable rule.
Sampling can draw whole clusters at a time (`--sampling clusters`) to
preserve the multi-well-per-cluster structure the planner faces in
production; independent per-well sampling mostly destroys it.

Each episode can also draw its own crew configuration, economic horizon,
work window and (optionally) an annual oil-production cap from configured
mixes, so the policy is trained over a *distribution* of planning regimes
rather than a single fixed one — this is what makes an out-of-the-box model
transfer to configurations it wasn't literally trained on. A mix is sampled
uniformly over its entries, so repeating one weights it: `bc39`'s
`--crew-mix 2x1,2x1,2x1,2x5,...` draws the tight `2x1` configuration three
times as often as the rest, because that is the regime it is most often asked
for. See `experiments/train.py`'s module docstring and `--help` for every
knob, and `models/bc39/manifest.json` for the exact distribution and
hyperparameters used to train the released model.
