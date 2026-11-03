# V1.3 visual FetchPush research record

Status: **CPU_VERIFIED** for actual PyTorch training and MuJoCo closed-loop execution; **NOT_RUN** for formal cloud-scale V1.3b confirmation. Both cameras are synchronized before each action, and training targets are the next four actions. The teacher is the privileged geometric `PushExpert`; the student receives two RGB frames at each of two observation times and, in multimodal variants, only 10 robot proprioceptive values. Object/goal coordinates are excluded from numeric student inputs. The [FetchPush task](https://robotics.farama.org/main/envs/fetch/push/) is a simulation manipulation benchmark, not LIBERO or a real robot.

## Preserved sensor failure, V1.3a

The first frozen protocol (`configs/v13/visual_fetch_push.json`, source commit `62959cc`) used MuJoCo's nominal head and gripper cameras. Post-run image inspection showed the head mostly viewed the robot shell/floor and the wrist view was dark. FetchPush's virtual desired goal was not rendered. This is a **sensor/task-specification failure**. All three visual models trained for 600 real updates on 48 teacher episodes and were evaluated on untouched seeds 92000–92019. They each scored **1/20**, and all succeeded only on seed 92012, whose initial object was already within the 5cm success threshold. Thus no policy showed a learned push success in this test. The descriptive state-only baselines scored ACT 10/20 and Diffusion 1/20, but used an older 64-episode dataset and different training budgets. Raw manifests, checkpoints and traces remain under `artifacts/v13/visual_fetch_push/`; no failed run was overwritten.

| V1.3a policy | Validation action MSE | Closed-loop success | Median policy inference per episode |
| --- | ---: | ---: | ---: |
| Vision-only ACT | 0.169 | 1/20 (initial success only) | 3.29ms |
| Vision + proprio ACT | 0.155 | 1/20 (initial success only) | 4.18ms |
| Vision + proprio DDPM | 0.958 | 1/20 (initial success only) | 9.51ms |

These timings exclude rendering and simulator stepping. Low validation MSE did not predict closed-loop success; the student could mimic action averages without seeing the target. Failure taxonomy: missing visible goal, unusable wrist camera, likely compounding action error and missed contact. The post-hoc diagnosis must not be used to retune and then reuse the same 20 seeds as independent confirmation.

## Corrected V1.3b visual contract and smoke

`task_visible` uses the fixed `external_camera_0`, a free camera following **gripper proprioception only**, and a red rendered goal marker. The student sees the marker as pixels, never numeric goal coordinates. Both views visibly include the tabletop, object and marker; the camera rig is a documented task augmentation and should not be described as the unmodified official FetchPush sensor configuration. Camera calibration and setup are in the dataset manifest. Actions remain bounded 4D Cartesian delta/gripper commands at 25Hz.

The local `v13b_task_visible_local_smoke_20261103` protocol used 8 training, 3 validation and 2 evaluation episodes, 30 updates per policy. It verified collection, learning gradients, checkpoint reload, inference, action MSE and closed-loop execution. All students scored **0/2**; this tiny smoke is not an estimate of final success. A separate four-seed smoke on 104000–104003 scored 0/4 for all models under nominal, 5cm fixed-camera shift and 3cm object shift alike. These are code-path checks, not usable generalization estimates. A replayed student failure on seed 103200 lasted 80 steps and ended at 0.166m goal distance; the privileged teacher succeeded in 16 steps at 0.046m. Their synchronized two-view videos and SHA256 metadata are under `artifacts/v13/visual_task_visible_smoke/videos/`. This comparison confirms the teacher/student capability gap; it does not show distillation success.

Formal `configs/v13/visual_fetch_push_task_visible_cloud.json` freezes 128 train, 24 validation and 30 confirmation episodes, 1500 updates per policy and training seed 41. It has **NOT_RUN**. Independent 10-seed nominal, 5cm fixed-camera-shift and 3cm object-position-shift evaluation is implemented in `worldpolicy.visual_evaluate` but likewise awaits trained cloud checkpoints. One training seed and a compact DDPM MLP would still be insufficient for an L2 multi-seed, strong-baseline claim. The cloud runbook specifies GPU sizing, hashes, resume and cost gates.
