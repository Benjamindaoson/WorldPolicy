# Official LIBERO / LeRobot integration

Status: **BLOCKED** for execution on this machine. `scripts/libero_linux_preflight.py` reported Windows, installed PyTorch/MuJoCo and absent `lerobot`, `libero`, `robosuite`. No usable Linux/WSL environment or GPU was available. No system components were changed. No official dataset, checkpoint or task was run.

The [official LeRobot LIBERO documentation](https://huggingface.co/docs/lerobot/main/en/libero) requires Linux, exposes `lerobot-eval`, and specifies 8D proprioception, main/wrist RGB cameras and a bounded 7D action. It supports `relative` and `absolute` action modes; these must match the selected checkpoint. Hard reset is the reference behavior. A FetchPush checkpoint here uses a **different** 28D/4D contract and cannot be renamed into a LIBERO policy.

Linux reproduction entry point: `scripts/libero_official_linux.sh`. It requires an explicit LeRobot commit, official checkpoint revision, suite, control mode and dataset revision. The script stops if the checkpoint observation schema/action mode is not manually verified; that verification cannot be inferred from a model name. Log the resolved upstream commit, dataset revision, model commit, camera keys, action mode, task IDs, reset setting, seed, max steps, and per-task success traces before using results in a report.

Planned suites: Spatial, Object, Goal, then Long. Planned policies: official ACT, official Diffusion, frozen SmolVLA, and fine-tuned SmolVLA only with suitable hardware. All are **NOT_RUN**. A frozen SmolVLA checkpoint may need task-specific adaptation, so a zero-shot attempt would be labeled separately from fine-tuned evaluation. The [official SmolVLA documentation](https://huggingface.co/docs/lerobot/smolvla) governs model and processor use.
