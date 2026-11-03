# V1.3 implementation and verification status

| Workstream | Status | Evidence / limit |
| --- | --- | --- |
| V1.2 scientific audit | CPU_VERIFIED | `results/v13/v12_audit.json`, paired 150-row CSV; only 50 unique evaluation initial states. |
| Dual-camera RGB data, timestamps, proprio and goal marker | CPU_VERIFIED | `visual_data.py`, manifest hashes, visible-camera tests, real MuJoCo frames. |
| Visual ACT/DDPM PyTorch updates and checkpoint loading | CPU_VERIFIED | V1.3a 600 updates × 3 models; V1.3b visible-task 30-update smoke × 3. Formal 1500-update cloud run NOT_RUN. |
| Visual closed-loop success | SIMULATION_VERIFIED failure | V1.3a all 1/20 from an initially successful seed; V1.3b smoke all 0/2. No learned visual push success established. |
| Visual success/failure video | SIMULATION_VERIFIED | Saved student failure and privileged teacher success on seed 103200, with matching replay and SHA256. |
| Independent OOD camera/object shift | CPU_VERIFIED smoke, NOT_RUN formally | Four fresh smoke seeds, 0/4 across conditions for each undertrained policy; `visual_evaluate.py` uses paired stochastic sampling. |
| Official LIBERO/LeRobot | BLOCKED | Linux unavailable locally; official weights and dataset not loaded. |
| Official SmolVLA | NOT_RUN | No official checkpoint or processor executed. |
| Robot dynamics and CEM | CPU_VERIFIED exploratory | V1.2 robot ensemble, physics oracle, learned CEM; no learned planning gain. |
| Model-based actor-critic | CPU_VERIFIED exploratory | 1,043 real transitions, 32,000 filtered imagined samples; 2/10 versus BC warmup 3/10. |
| Corrective learning/distillation | NOT_RUN | No corrective checkpoint or world-planner student. |
| SO-101 adapter/motion | BLOCKED | `RobotGate` is fail-closed; no camera/state/action hardware transport or torque command. |
| WorldSystems policy contract | TESTED on smoke checkpoint | JSON schema, checkpoint SHA, 2-view input and bounded 4D inference through `policy_interface.py`; no other repository modified. |
| Local cloud preparation | TESTED / GPU_SIZING_REQUIRED | Exact Linux dependency lock, GPU probe, budget/monitor/backup scripts; CUDA execution awaits authorized instance. |

## Acceptance assessment

| Level | Result | Reason |
| --- | --- | --- |
| L1 | **PASS for state-only FetchPush simulation scope** | V1.2 real MuJoCo operation, trained checkpoint reload, closed-loop evaluation, video and tests remain valid. This does not mean visual policy success or real-robot verification. |
| L2 | **NOT_MET** | Visual student has no demonstrated learned success; no official LIBERO result, multi-seed visual confirmation or reliable world-model planning benefit. Compact MBRL exists but its single-seed result is negative. |
| L3 | **NOT_MET** | No validated novel method, cross-benchmark confirmation or paper-level downstream result. |

Hardware: Windows GTX 1650 4GB, CPU-only local PyTorch, no usable Linux LIBERO environment. GPU_VERIFIED and REAL_ROBOT_VERIFIED are **not claimed**. All new runs use unique ignored `artifacts/v13` paths; original V1–V1.2 artifacts remain intact. No remote push, cloud rental or robot motion occurred. See `CLOUD_MIGRATION_STATUS.md` for the next paid-resource gate.
