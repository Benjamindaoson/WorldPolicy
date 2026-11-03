# Policy baselines

Status: ACT and Diffusion IMPLEMENTED / TESTED / CPU_VERIFIED on MuJoCo reach; official LeRobot ACT and SmolVLA NOT_RUN. Both local policies used the same 64 expert episodes, 4-step action chunks, 500 optimizer updates, batch size 64, learning rate 5e-4, seed 11, and identical 12 evaluation seeds. Architectures and inference costs differ, so compute budgets were not equal. Both receive state only; RGB is not fed into either model.

| Policy | Validation action MSE | MuJoCo successes | Wilson 95% |
| --- | ---: | ---: | --- |
| Local ACT-style CVAE Transformer | 0.4811 | 9/12 | [0.468, 0.911] |
| Local conditional DDPM | 0.5506 | 3/12 | [0.089, 0.532] |
| Jacobian expert | not trained | 12/12 | [0.757, 1.000] |

Paired ACT/Diffusion outcomes: both success 1, ACT only 8, Diffusion only 2, both fail 1. The sample is too small and one training seed cannot establish algorithm superiority. The local ACT and DDPM are not architecture-equivalent to official LeRobot implementations. Frozen SmolVLA was not loaded or evaluated. See `artifacts/v11/mujoco_reach/summary.json` and each policy's `evaluation.json` and checkpoint for exact traces.
