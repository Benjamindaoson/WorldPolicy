# WorldPolicy V1.2 implementation and execution status

| Workstream | Status | Verifiable evidence |
| --- | --- | --- |
| V1.1 audit and checkpoint/video identity | TESTED | `V12_V11_AUDIT.md`; original V1/V1.1 artifacts retained |
| Official FetchPush-v4 contact simulator | CPU_VERIFIED | `fetch_push.py`, tests, 20-seed expert preflight |
| Frozen V1.2a ACT/DDPM baseline | CPU_VERIFIED | `artifacts/v12/fetch_push/summary.json`; failure preserved |
| Frozen V1.2b ACT/DDPM, 3 seeds × 50 initial states | CPU_VERIFIED | `artifacts/v12/fetch_push_b/summary.json`; per-policy checkpoint/evaluation JSON |
| Success and failure replay videos | CPU_VERIFIED | `artifacts/v12/fetch_push_b/seed_21/{act,diffusion}/*.mp4` and matching JSON |
| Approximately equal CPU-time control | CPU_VERIFIED | `artifacts/v12/fetch_push_compute_control/evaluation.json` |
| Physics-oracle random shooting / CEM | CPU_VERIFIED | `artifacts/v12/fetch_push_oracle_10.json` |
| Robot-conditioned dynamics and learned CEM | CPU_VERIFIED | `artifacts/v12/fetch_push_dynamics/`, `fetch_push_learned_10.json` |
| Friction-shift OOD, paired 50 states × 3 training seeds | CPU_VERIFIED | `artifacts/v12/fetch_push_ood.json` |
| Official LIBERO / LeRobot | BLOCKED | Windows without usable WSL Linux or official stack |
| Official SmolVLA | NOT_RUN | No official weights loaded or official benchmark setup |
| Full model-based RL, CQL, corrective SFT, distillation | NOT_RUN | No imagined actor/critic update |
| SO-101 real movement | BLOCKED | Safety gate remains closed |

Commands from repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,manipulation]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\python.exe -m worldpolicy.fetch_push_experiment --config configs/v12/fetch_push_protocol_b.json --output artifacts/v12/fetch_push_b
.\.venv\Scripts\python.exe -m worldpolicy.fetch_push_compute_control
.\.venv\Scripts\python.exe -m worldpolicy.fetch_push_dynamics
.\.venv\Scripts\python.exe -m worldpolicy.fetch_push_planning --learned-checkpoint artifacts/v12/fetch_push_dynamics/checkpoint.pt
.\.venv\Scripts\python.exe -m worldpolicy.fetch_push_ood
.\.venv\Scripts\python.exe scripts/record_fetch_push_cases.py
```

Source, configs, tests and concise reports are committed locally. Large checkpoints, transition arrays and videos remain ignored under `artifacts/v12/`. No remote push or real robot operation occurred.

## Acceptance gates

| Level | V1.2 assessment | Reason |
| --- | --- | --- |
| L1 Research Engineer | PASS for state-only FetchPush simulation chain | Standard robot/object contact task, saved demonstrations, ACT/DDPM updates and checkpoints, independent paired evaluations, tests and raw traces. This is a simulation engineering result, not a visual VLA or real-robot result. |
| L2 Strong Research Engineer | PARTIAL | Three training seeds, frozen confirmation, friction OOD and robot-conditioned planning exist; official LIBERO/SmolVLA and reliable learned-model benefit do not. |
| L3 Research Scientist | NOT_MET | No original method with confirmatory cross-task benefits or independent replication. |
