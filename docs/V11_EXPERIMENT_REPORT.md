# V1.1 execution report — 2026-11-03

All results are **SIMULATION** on local CPU PyTorch 2.15.0. MuJoCo 3.17.0 ran a custom two-link arm reach task; point-reach diagnostics are separate. Source commits: audit fixes `87c64e2`, MuJoCo/diagnostics `a7d40cd`. Raw artifacts are ignored by Git and retained locally under `artifacts/v11/` and `results/v11/`.

| Run | Data / updates | Test seeds | Success |
| --- | --- | --- | ---: |
| MuJoCo ACT | 64 demonstrations / 500 updates | 2011–2022 | 9/12 |
| MuJoCo Diffusion | same 64 demonstrations / 500 updates | 2011–2022 | 3/12 |
| MuJoCo Jacobian expert | analytic, no training | 2011–2022 | 12/12 |
| Point-reach learned CEM | V1 learned ensemble; 64 random-action training episodes | 2000–2007 | 5/8 |
| Point-reach oracle CEM | exact physics, no training | 2000–2007 | 7/8 |
| Point-reach learned CEM horizon 4 | same learned ensemble, exploratory tuning | 2000–2007 | 8/8 |
| Corrected point-reach IQL | 24 expert episodes / 100 updates | 1007–1014 | 3/8 |

The CEM horizon-4 result was selected on the diagnostic cases and needs fresh-seed validation. V1 IQL's 5/8 was based on incorrect terminal labeling; do not compare it as a valid IQL result. MuJoCo ACT/Diffusion differ in training wall time (about 18.1 s versus 1.1 s) and cannot support an equal-compute conclusion. No GPU training or actual robot experiment occurred. No LIBERO benchmark or official VLA result exists.
