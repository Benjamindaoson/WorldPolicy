# V1.1 status and continuation

| Component | Status | Evidence / next gate |
| --- | --- | --- |
| V1 audit and fixes | TESTED | `V11_V1_AUDIT.md`; 15+ pytest cases |
| MuJoCo arm reach and RGB renderer | CPU_VERIFIED | `mujoco_reach.py`, `assets/two_link_reach.xml` |
| MuJoCo ACT / Diffusion checkpoints | CPU_VERIFIED | `artifacts/v11/mujoco_reach/{act,diffusion}/` |
| Point-reach CEM diagnosis | CPU_VERIFIED | `artifacts/v11/point_reach_diagnostics.json` |
| Corrected IQL replay | CPU_VERIFIED | `results/v11/iql.json`, `iql_replay.npz` |
| MuJoCo learned-world-model planning | NOT_RUN | Train compatible arm dynamics and validate on held-out transitions |
| Official LIBERO / LeRobot | BLOCKED | Linux environment and benchmark assets absent |
| SmolVLA / OpenPI | NOT_RUN | Official model and compatible Linux setup absent |
| Model-based RL / CQL / distillation | NOT_RUN | Requires validated MuJoCo dynamics and policy-update implementation |
| SO-101 motion | BLOCKED | Safety gate closed; no hardware authorization |

Reproduce tests: `.\\.venv\\Scripts\\python.exe -m pytest -q` and `.\\.venv\\Scripts\\ruff.exe check .`. Reproduce MuJoCo run: `.\\.venv\\Scripts\\python.exe -m worldpolicy.mujoco_experiment`. Reproduce point-reach diagnosis: `.\\.venv\\Scripts\\python.exe -m worldpolicy.diagnostics`. Outputs are local and never pushed automatically.

Priority continuation: (1) fresh-seed confirmation of horizon-4 diagnostic, (2) train MuJoCo arm dynamics and matched base/oracle/learned policy controls, (3) Linux LIBERO setup and official data/weight validation, (4) model-based policy updates and multi-seed study.
