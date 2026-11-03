# CEM failure analysis

Status: CPU_VERIFIED on point-reach, eight paired initialization seeds 2000–2007. Code: `src/worldpolicy/diagnostics.py`; raw trajectories: `artifacts/v11/point_reach_diagnostics.json`; learned dynamics SHA256 `f4de39d018b77fb93f5f3afd595af6dbf5219d3c04c8ac920eb4301f1cc4de1c`. This is exploratory diagnosis, not a tuned held-out benchmark.

| Planner | Successes / 8 | Mean return |
| --- | ---: | ---: |
| Random action | 0 | -63.93 |
| Analytic expert | 8 | -9.97 |
| Random shooting, exact physics | 4 | -25.75 |
| CEM, exact physics | 7 | -16.71 |
| Random shooting, learned dynamics | 3 | -26.39 |
| CEM, learned dynamics | 5 | -17.38 |
| Learned CEM, uncertainty penalty 10 | 0 | -20.55 |
| Learned CEM, horizon 4 instead of 8 | 8 | -12.05 |

Prediction state MSE by horizon: 1: `5.80e-6`; 2: `1.97e-5`; 4: `7.40e-5`; 8: `3.16e-4`. Candidate ranking correlation at horizon 8 was 0.998; mean absolute goal-cost error on the selected learned plan was 0.00914, versus exactly zero for the physics oracle. The small random-candidate ranking error does not eliminate local optimizer bias or closed-loop compounding.

**Interpretation:** The oracle's 7/8 versus expert's 8/8 shows planning/optimization limitations even with exact dynamics. The learned planner's 5/8 adds model error. The terminal cost at eight control steps asks the optimizer to solve an often unreachable short horizon. Shortening to four steps led to 8/8 on these same eight cases, but it was selected after viewing failures and must be validated on fresh seeds. A weight-10 uncertainty penalty was harmful; ensemble variance is not yet calibrated to task risk. The older V1 2/8 result used a different population, planner RNG trajectory, and protocol, so its change to 5/8 cannot be attributed to one fix.

Next diagnostic gate: fresh paired seeds and task variations, calibrated uncertainty, horizon/population sensitivity, and independently trained dynamics on MuJoCo. Do not claim learned-world-model benefit from this point-reach study.
