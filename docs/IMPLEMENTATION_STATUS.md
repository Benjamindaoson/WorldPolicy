# Implementation and execution status

This file separates code from experimental evidence. `EXECUTED` means a local process completed and wrote an auditable JSON record; it does not imply robot manipulation success.

| Area | Status | Evidence / reason |
| --- | --- | --- |
| ACT / DDPM training and checkpoint | EXECUTED | `artifacts/{act,diffusion}/train.json` |
| Point-reach closed-loop evaluation | EXECUTED | `artifacts/{act,diffusion}/evaluation.json` |
| Physics-oracle CEM | EXECUTED | `results/planning.json` |
| Learned dynamics ensemble and paired CEM | EXECUTED | `artifacts/dynamics/{train,evaluation}.json` |
| IQL continuous offline update | EXECUTED | `results/iql.json` |
| MuJoCo / LIBERO | NOT_RUN | Dependencies and benchmark assets not provisioned |
| Learned WorldDynamics planning | BLOCKED | No exported, validated checkpoint contract |
| SmolVLA / OpenPI training | NOT_RUN | Official weights and accelerator budget not provisioned |
| Model-based RL / CQL / corrective / distillation | NOT_RUN | Implementation pending |
| SO-101 motion | BLOCKED | Safety gate closed; calibration and human approval required |

## Local CPU run, 2026-11-03

All numbers below are from this repository's small point-reach physics baseline. They are not LIBERO, MuJoCo, manipulation, or physical robot results. The first two policies used 32 generated expert episodes and seed 7; each was evaluated on eight held-out initialization seeds (1007–1014). The dataset and evaluation environment share the same task definition. Checkpoints and full per-episode records remain in ignored local artifact directories.

| Run | Update steps | Held-out action MSE | Closed-loop success |
| --- | ---: | ---: | ---: |
| ACT | 120 | 0.00974 | 3/8 |
| Diffusion Policy | 300 | 0.16739 | 3/8 |
| IQL | 100 | Not measured | 5/8 |

The learned dynamics ensemble was trained on 64 random-action point-reach episodes for 500 steps per member. Its one-step MSE on eight independent environment seeds was 0.0000799. For **paired** closed-loop initialization seeds 2000–2007, learned-dynamics CEM succeeded 2/8, physics-oracle CEM 5/8, and the analytic expert 8/8. This is a negative result for the current learned planner. The oracle uses exact environment dynamics and is not a learned-world-model baseline. The separate `results/planning.json` run used different seeds and must not be compared directly to this paired experiment.

The eight episodes per condition are too few for a research claim. Training datasets and compute budgets differ across IQL and the planners, so those success rates are descriptive only. There is no confidence interval or causal world-model benefit conclusion.

## Engineering verification

`pytest -q`: 8 passed. `ruff check .`: passed. Editable package install succeeded in the local Python 3.12 virtual environment. CPU PyTorch 2.15.0 was used; CUDA, MuJoCo, and LIBERO were not available in the initial environment. GitHub cloning failed due to network connection failure, so the repository was initialized locally with the intended remote URL.

The point-reach environment uses known Euler dynamics. Its oracle result must never be reported as a learned-world-model benefit. Formal comparisons require matched episodes, seeds, budgets and independent held-out physical simulation.
