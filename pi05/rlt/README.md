# RLT learner core (`pi05/rlt/`)

Stage 2 of the Cart FSD π0.5 program: **RL-Token (RLT)** online RL (Xu et al.,
Physical Intelligence). The fine-tuned π0.5 stays **frozen**; a small
actor-critic — conditioned on a compact "RL token" `z_rl` extracted from π0.5 —
learns to *locally edit* π0.5's native `(accel, curvature)` reference action
chunk, anchored to it by a BC regularizer, from **sparse binary human
success/fail labels** plus **human (PS5) interventions**.

This directory is the **learner core only**: the framework-agnostic,
fully-testable RL machinery. Everything that touches real π0.5 weights or
hardware is **deferred behind ABC seams** (see [Seam contract](#seam-contract))
until the friend's 10 Hz π0.5 Thor inference code lands.

## State / action

- State `x = (z_rl ∈ R^token_dim, s_p ∈ R^dim_proprio)`. `z_rl` is opaque (the
  RL token); `s_p` is proprioception from a high-quality streamed egomotion
  source. Both are supplied through interfaces — the core never assumes their
  internals.
- Action `a ∈ R^{C×d}`, `d=2` = `(acceleration m/s², curvature rad/m)`, `C=5`
  @ 10 Hz. The actor sees `(z_rl, s_p, ã[:C])` where `ã` is π0.5's reference
  chunk, and outputs an *edited* chunk squashed into π0.5's trained bounds.

## Math

**Actor** `π_θ(a | z_rl, s_p, ã) = N(μ_θ, σ²I)`, fixed per-dim σ (no log-σ head,
no entropy term). `μ_θ` is `tanh`-squashed per dim into `[low, high]`, so every
proposal is kinematically valid regardless of `ã` scale. `forward` returns the
deterministic mean (the export/inference path); `rsample` reparameterizes for
training.

**Critic** `QEnsemble`: N ≥ 2 Q-nets with Polyak targets; `min`-of-N for both
the TD target and the (pessimistic) policy objective.

**Chunked C-step TD target** (computed in `losses.critic_td_target`):

```
y = Σ_{k=0..C-1} γ^k · r_k · d_k  +  bootstrap · γ^n · min_i Q'_i(x', a'),
    a' ~ π_target(· | x', ã')
```

`bootstrap`, `gamma_pow = γ^n`, the per-step `rewards`, and the `discounts` mask
are **precomputed once at insert time in the buffer** and consumed verbatim —
the loss never re-derives them.

**Policy loss** (RLT Eq. 5):

```
L_π = mean( -min_i Q_i(x, a)  +  β · ‖a − ã‖² ),   a = rsample(μ_θ(x, ã_in), σ)
```

`β` is the BC-anchor / stability knob; `γ` is the speed knob (sparse terminal
reward + discount ⇒ finishing sooner ⇒ higher return).

### Two subtle, deliberate decisions

1. **Reference-dropout asymmetry.** In the **policy loss only**, `ã_in` is zeroed
   per-sample w.p. `ref_dropout_p` to force the actor to learn an independent
   pathway instead of copying `ã`. It is **never** applied to the TD target's
   `a'`: the deployed actor always sees `ã`, so the value backup must too. The
   BC anchor always pulls toward the **true** `ã`.
2. **Timeout ≠ terminal.** `bootstrap = 0` only when a sub-chunk lands on a
   `SUCCESS`/`FAILURE` terminal. Interior chunks **and** `TIMEOUT` bootstrap
   (a time limit doesn't end the MDP — `V(x')` is still meaningful). Decided
   once in the buffer, unit-tested there.

## Replay buffer (`buffer.py`) — highest correctness risk

Two-phase because terminal reward is unknown until the episode ends:

1. `add_step(...)` stages one raw per-timestep record per `episode_id`, carrying
   the current state, the **one-step-ahead** next state (so a boundary sub-chunk
   still has a real bootstrap state — needed for TIMEOUT), the immediate reward
   (≈ always 0), and source/intervention flags.
2. `finalize_episode(episode_id, done_type, terminal_reward)`:
   - backfills the terminal reward (only `SUCCESS` → `terminal_reward` at the
     last step; `FAILURE`/`TIMEOUT`/`NONE` stay sparse);
   - rewrites intervention rows so `a_ref = a` (the human action is both
     executed and the BC reference);
   - emits **stride-subsampled** C-step sub-chunks. For each `t0`:
     `n = min(C, T−t0)`, post-`n` steps masked via `discounts`, `x'` is the
     next-state of the last executed step, `bootstrap`/`gamma_pow`/`done_type`
     set as above. Every boundary tail with `n ≥ 1` is kept (short tails carry
     the only nonzero rewards).

Multi-source sampling is **uniform** by default; `cfg.source_weights` opts into
per-source upweighting (e.g., demo + intervention). A staging watchdog
(`flush_stale`) finalizes never-closed episodes as `TIMEOUT`.

## Learner (`learner.py`)

`step()` runs `utd` critic updates with a `critic_per_actor:1` critic:actor
cadence, Polyak after every online update, gated by `learning_starts`. The
**actor only** is periodically exported via an atomic file swap
(`export.py`) so the runtime hot-reloads weights with no cross-thread tensor
races. `start()/stop()` run the loop on a background thread (a lock serialises
buffer access so a rollout thread can `add_step` concurrently).

## Safety (`safety.py`)

`NativeUnitsSafetyLayer` clamps in π0.5's **native units**: absolute bounds
(accel ∈ [−9.8, 9.8], curvature ∈ [−0.2, 0.2]), per-step rate caps (jerk /
curvature-rate), and a training speed cap.

> **Unit boundary.** `limits.py` (`effective_gas_cap`, `STEERING_*`) is the
> single source of truth for the **actuation** clamp, but it speaks normalized
> pot / steering-degree units — it applies at the deferred **tracker seam**
> (accel→gas-pot, curvature→steering-angle), **not** here. Native units in the
> core, command units at the tracker; the two clamps compose without
> double-converting.

## Seam contract

The deferred on-vehicle pieces implement these ABCs (`interfaces.py`):

| ABC | Method | Deferred concrete implementation |
|-----|--------|----------------------------------|
| `VLAWrapper` | `encode(obs) -> (z_rl[token_dim], a_ref[C,d])`, `.token_dim`, `.chunk_len` | Friend's frozen π0.5 forward + the RL-token extractor over the `prefix_out` embedding hook (already computed at openpi `pi0.py:209`, just discarded today). |
| `EgomotionStream` | `read() -> s_p[dim]`, `.dim` | Streamed high-quality egomotion readout. |
| `RewardInterface` | `evaluate(obs) -> (reward, DoneType)` | Human success/fail label at episode end. |
| `SafetyLayer` | `clamp(a, state) -> (safe_a, was_modified)` | `NativeUnitsSafetyLayer` (concrete now); actuation-unit clamp via `limits.py` lives at the tracker. |

`envs/unicycle.py` ships `SyntheticVLA` / `SyntheticEgomotion` stand-ins so the
whole stack is exercised against the *real* interfaces today.

## Verify

```bash
uv run pytest pi05/rlt/tests/ -q       # unit tests (buffer / targets / nets / safety)
uv run python -m pi05.rlt.validate     # end-to-end "learning curve climbs" gate
```

## Deferred (next pass, when the friend's Thor code lands)

RL-token encoder/decoder + reconstruction trainer (`L_ro`); concrete
`VLAWrapper` over the friend's π0.5 forward + `prefix_out` hook;
waypoint→ODrive/pedal tracker (where `limits.py` clamps apply); PS5-intervention
control loop; Thor control node + weight-sync transport.
