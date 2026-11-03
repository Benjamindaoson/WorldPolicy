# WorldPolicy — World-Model-Guided Robot Learning and Policy Improvement

WorldPolicy is an auditable research codebase for continuous robot policy learning and world-model-assisted control. Its established manipulation results use **privileged state input in MuJoCo Gymnasium-Robotics `FetchPushDense-v4`**. They are simulation results, not official LIBERO, language-conditioned VLA, or real-robot results.

## What runs

- ACT-style CVAE Transformer and conditional DDPM Diffusion Policy, with action chunks, checkpoint save/load, and closed-loop evaluation.
- Continuous-action IQL with expectile value learning, double Q targets, and advantage-weighted actor updates. Its corrected small-task result is separate from FetchPush; there is **no FetchPush IQL result**.
- Versioned demonstrations, action contracts, episode splits, SHA256 dataset validation, and replayable evaluation traces.
- Actual Gymnasium-Robotics FetchPush contact simulation, robot-conditioned learned dynamics, physics-oracle MPC/CEM, and learned-dynamics CEM.
- Experimental two-camera RGB ACT/DDPM policies, a fail-closed robot safety gate, and a compact model-based actor-critic baseline. Current visual and model-based RL results are negative; see the reports below.

## Measured FetchPush results

| Method | Evaluation | Outcome |
| --- | --- | --- |
| State-only ACT | Three training seeds, 50 paired initial states each | **27/50, 35/50, 33/50** |
| State-only compact Diffusion Policy | Same demonstrations and paired initial states within each seed | **5/50, 6/50, 11/50** |
| Learned robot-dynamics CEM | 10 exploratory initial states | **3/10** |
| MuJoCo physics-oracle CEM | Same 10 exploratory initial states | **10/10** |
| V1.3 first visual policies | 20 held-out initial states | **1/20 each; the sole success was already within the goal threshold at reset** |
| V1.3 compact model-based actor-critic | 10 exploratory initial states | **2/10**, versus BC warmup **3/10** |

These protocols have different training data, compute budgets, and test sets unless explicitly paired. The physics oracle simulates counterfactual actions with MuJoCo snapshots and is not a learned planner. A [scientific audit](docs/V13_SCIENTIFIC_AUDIT.md) found that the three V1.2 training seeds share the same 50 evaluation initial states, so 150 records do not mean 150 independent scenes. The first RGB run had an unusable camera/goal view; corrected cameras passed a local smoke test but have **not** passed full visual confirmation. No claim that Diffusion Policy is generally weaker than ACT follows from these compact baselines.

## Reproduce locally

Python 3.12 is used for the verified Windows environment. Run from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,simulation,manipulation]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\python.exe -c "from worldpolicy.fetch_push import FetchPush; e=FetchPush(1); print(e.reset().shape); e.close()"
.\.venv\Scripts\python.exe -m worldpolicy.v13_audit
```

The short visual smoke actually trains three models and evaluates two fresh resets; it is **not** a success-rate benchmark:

```powershell
.\.venv\Scripts\python.exe -m worldpolicy.visual_experiment --config configs/v13/visual_fetch_push_task_visible_smoke.json --output artifacts/v13/my_visual_smoke --device cpu
```

The longer frozen V1.2 FetchPush protocol is `configs/v12/fetch_push_protocol_b.json` and its CLI is `python -m worldpolicy.fetch_push_experiment --config ... --output ...`. It retrains models and should be run in a new output directory. Large historical checkpoints, datasets, videos and per-episode traces are kept locally under ignored `artifacts/` and `results/`; the public repository carries their protocols and honest summaries, not those binary files.

## Research status and cloud work

The [V1.3 report](docs/WORLDPOLICY_RESEARCH_REPORT.md) links the visual, diffusion, planning and model-based RL evidence. [Implementation status](docs/V13_IMPLEMENTATION_STATUS.md) distinguishes tested simulation from unrun work. Official [LIBERO/LeRobot](docs/V13_LIBERO_INTEGRATION.md) and SmolVLA are **blocked or not run** on the local Windows CPU-only PyTorch setup. No real-robot motion was attempted. The [cloud runbook](docs/CLOUD_GPU_RUNBOOK.md) provides an isolated Linux environment, hash-pinned dependencies, a real CUDA preflight, resumable training, cost controls, and verified backup; paid resources still require authorization.

External simulators, datasets, checkpoints, and pretrained models retain their own upstream licenses. They are not bundled here. This repository does not currently declare a license for reuse of its own code.
