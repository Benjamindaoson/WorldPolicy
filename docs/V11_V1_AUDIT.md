# V1 audit (2026-11-03)

Status: TESTED / CPU_VERIFIED. Source baseline: `c316bcd`. Fixes: `87c64e2`. Original V1 artifacts remain under `artifacts/{act,diffusion,dynamics}` and `results/`.

| Area | Finding | V1.1 action |
| --- | --- | --- |
| Dataset | V1 ACT/Diffusion used disjoint episode indices (25 train, 7 held-out action test) from 32 seeded point-reach demonstrations. No separate validation set or initial-state hash audit. Point-reach observations were state-only, not RGB. | MuJoCo experiment uses 64 demonstrations, disjoint episode validation subset, and separate evaluation seeds 2011–2022. |
| ACT | Genuine action chunks, CVAE KL/reconstruction and optimizer steps. Posterior transformer lacked action position embeddings; it could not reliably encode action order. It is a small state-only ACT-style model, not the official visual ACT architecture. | Learned action position embeddings added; new checkpoints required. |
| Diffusion | Forward noise and reverse denoising both ran, but 20-step linear beta left alpha-bar about 0.82 at the endpoint although sampling started from pure Gaussian noise. Reverse step used beta rather than posterior variance. | Cosine schedule reaches near zero alpha-bar; posterior variance used. Old checkpoint is not compatible with the corrected schedule. |
| Checkpoints | Model and optimizer weights saved; no supported resume or RNG state. | Resume restores optimizer and Python/NumPy/Torch RNG states. Tested. |
| IQL | Expectile value, double Q, Bellman target, advantage-weighted actor and target updates were present. Replay construction omitted final transitions and marked the penultimate one done regardless of actual success. | Every physical step records reward, next state, success terminal, and timeout separately. V1 5/8 result is invalidated for method comparison; V1.1 rerun is 3/8 on different seeds. |
| CEM | Bounded, iterative elite optimization was genuine. Eight-step terminal distance ignores interim reward and cannot generally reach the target from the starting state. V1 shared planner RNG across episodes; subsequent variants did not receive identical candidate streams. | Paired diagnostic resets planner RNG per seed and records candidate controls, model error, and closed-loop outcomes. |

The original point-reach successes are descriptive software evidence only. V1 had no MuJoCo manipulation benchmark or learned-model downstream improvement claim. No real robot ran. The audit did not inspect a full upstream ACT/Diffusion equivalence test; this implementation should be cited as an independent small baseline.
