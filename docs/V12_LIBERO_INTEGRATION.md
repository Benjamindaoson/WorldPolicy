# Official LIBERO / LeRobot

Status: BLOCKED locally / NOT_RUN. Read-only preflight finds Windows without a usable WSL Linux distribution and without `lerobot`, `libero`, or `robosuite`. [Official LeRobot LIBERO guidance](https://huggingface.co/docs/lerobot/libero) requires Linux and specifies 8D proprioception, two camera views, and 7D end-effector/gripper actions. FetchPush's 28D state and 4D actions are incompatible; no padding/truncation adapter is allowed.

The project retains `scripts/libero_linux_preflight.py`. A future Linux experiment must pin LeRobot/LIBERO commits, dataset revision, policy checkpoint, camera keys, relative/absolute control mode, task IDs, resets and seeds before downloading or training. The official `lerobot/libero` dataset is not present. No LIBERO task or success metric was executed in this version.
