# WorldPolicy

World-model-guided robot learning and policy improvement. This repository contains **small research baselines**, including an actual MuJoCo two-joint arm reach simulation. It does not contain a validated manipulation benchmark result.

## Research question

Can learned world models improve robot policies under fixed data, compute, and interaction budgets? A future answer requires a learned model checkpoint, paired held-out tasks, and multiple seeds. No such claim is made here.

## Implemented and executed locally

- Versioned continuous-action episode format, SHA256 manifest, episode splitting, and action chunks.
- Trainable ACT-style CVAE Transformer and conditional DDPM policy with checkpoint reload.
- Deterministic continuous point-reach physics task with expert demonstrations and closed-loop evaluation. **This is a lightweight physics baseline, not MuJoCo, LIBERO, or a real robot.**
- World-model adapter contract, strict action specification check, physics-oracle CEM planning.
- Trained point-reach dynamics ensemble, uncertainty estimate, and a paired held-out CEM comparison. The learned planner underperformed the physics oracle in this small run.
- Continuous-action IQL with expectile value regression, double Q targets, and advantage-weighted actor update.
- Fail-closed real-robot gate. There is no motor-control transport in this version.
- V1.1: custom MuJoCo two-joint reach, paired ACT/Diffusion training and evaluation, corrected diffusion schedule and IQL replay, checkpoint resume, and CEM failure diagnostics. See [V1.1 report](docs/V11_EXPERIMENT_REPORT.md).

## Reproduction

Python 3.11+ is required. From this repository:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/train_act.py --config configs/policies/act.yaml
.\.venv\Scripts\python.exe scripts/train_diffusion.py --config configs/policies/diffusion.yaml
.\.venv\Scripts\python.exe -m worldpolicy.cli plan
.\.venv\Scripts\python.exe -m worldpolicy.cli train-iql
.\.venv\Scripts\python.exe -m worldpolicy.cli train-dynamics
.\.venv\Scripts\python.exe -m pip install -e ".[simulation]"
.\.venv\Scripts\python.exe -m worldpolicy.mujoco_experiment
.\.venv\Scripts\python.exe -m worldpolicy.diagnostics
```

Training artifacts and evaluation episode records are generated under ignored `artifacts/` and `results/`. They are local evidence and are not committed as scientific results. See `docs/IMPLEMENTATION_STATUS.md` for actual execution status and limitations.

## Architecture and boundaries

All actions carry embodiment, dimension, unit, frame, frequency, and bounds. Planning rejects a mismatched contract. The physics oracle is labeled separately from a learned world model. WorldDynamics integration requires a published checkpoint and versioned adapter. WorldSystems integration can consume checkpoint and evaluation metadata without sharing writable state.

## Research roadmap

Next: official LIBERO integration on Linux, paired OOD evaluations, MuJoCo arm dynamics and planning, model-based RL, corrective-data controls, official SmolVLA/OpenPI adapters, and supervised SO-101 commissioning. These are **not implemented or validated** in V1.1.
