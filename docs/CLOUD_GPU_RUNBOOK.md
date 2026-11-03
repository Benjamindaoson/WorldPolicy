# WorldPolicy cloud GPU runbook

All commands below refer to **WorldPolicy only**. Status: scripts are locally syntax-checked; Linux CUDA execution is **NOT_RUN**. The [resource gate](GPU_RESOURCE_REQUIREMENTS.md) gives measured local facts and a provisional quote. A paid Pod needs explicit platform, instance and maximum-cost authorization before creation.

## 1. Transfer the actual local branch

After all local changes are committed, run `scripts/cloud_make_bundle.ps1 -OutputDirectory artifacts/cloud_transfer` on Windows. Verify `bundle_identity.json` and SHA256. Transfer `worldpolicy-current.bundle` and the JSON by an approved SSH channel. On Linux, `sha256sum` must match, then run `git clone -b develop/worldpolicy-v1.3 worldpolicy-current.bundle WorldPolicy`. The `-b` is necessary because this single-branch bundle has no remote `HEAD`; a local clone test confirmed it. Verify the checkout HEAD against `bundle_identity.json`. Do not clone remote `main` as a substitute: it may lack local commits. No script pushes to GitHub.

Transfer datasets/checkpoints separately to approved private storage. Before transfer, use `python scripts/cloud_verify.py create --root artifacts/v13/visual_task_visible_smoke --manifest artifacts/cloud_transfer/smoke_sha256.json`; after transfer use `verify` with the same manifest. For the formal experiment, the cloud process can collect its own deterministic 128 training and 24 validation episodes from the frozen config. Preserve the original V1.2 artifacts locally; send only required checkpoints such as `artifacts/v12/fetch_push_b/seed_21/{act,diffusion}/checkpoint.pt`, `train.json`, `state_normalization.npz` if running the descriptive state baselines. Their SHA256 values must be checked against source records. No private keys, unrelated data or third-party weights belong in the transfer.

## 2. Isolated environment and CUDA smoke

On the authorized Linux instance, mount a persistent volume, set `WORLD_POLICY_VENV=/workspace/worldpolicy-venv`, and run `bash deploy/cloud/setup_linux.sh` from the repository root. The hash-pinned lock is `deploy/cloud/requirements-linux.lock` for Python 3.12 Linux. Run `python -m pytest -q` and `ruff check .` inside that venv. Check `nvidia-smi` and `torch.cuda.is_available()`; a CPU-only wheel fails the next gate.

Create a fresh persistent `RUN_DIR`. Run `"$WORLD_POLICY_VENV/bin/python" -m worldpolicy.visual_experiment --config configs/v13/visual_fetch_push_task_visible_cloud.json --output "$RUN_DIR" --collect-only` to generate the **actual formal** 128/24-episode train/validation datasets. Set `TRAIN_DATASET="$RUN_DIR/train_dataset"`, then run `bash scripts/cloud_benchmark.sh`. This executes five real forward/backward/optimizer steps per model at batch 32 and writes `gpu_preflight.json` plus GPU identity. Confirm peak allocated/reserved VRAM, per-step time, finite loss and GPU utilization. The present trainer uses float32; mixed precision should be tested and recorded separately before relying on it. Do not multiply parameter bytes to guess peak VRAM.

## 3. Budget and formal visual training

Verify the live provider checkout quote and storage charge. Record GPU model/count, observed peak VRAM, exact hourly all-in charge, projected runtime and authorized maximum. `cloud_train.sh` requires `CLOUD_RUN_AUTHORIZED=yes`, `GPU_HOURLY_USD`, `MAX_SPEND_USD`, `TRAIN_DATASET`, `RUN_DIR`, and `WORLD_POLICY_VENV`. It checks the measured preflight and train dataset hash, estimates cost at 2× measured optimizer time for nontraining overhead, and enforces a wall-clock timeout derived from the approved cap. It creates no instance and cannot increase a cap.

`bash scripts/cloud_train.sh` runs the actual `worldpolicy.visual_experiment --config configs/v13/visual_fetch_push_task_visible_cloud.json --device cuda --resume` CLI against those verified datasets. It trains three policies, validates actions and evaluates 30 held-out initial states. A restart uses `bash scripts/cloud_resume.sh` with the same variables and `RUN_DIR`; progress checkpoints are saved every 100 updates, and completed policy evaluations are skipped.

Monitor `logs/gpu_usage.csv`, `logs/cost.jsonl`, `logs/train.log`, each policy's `progress.pt` and `train.json`. The timeout may terminate between checkpoints; resume only within a newly authorized remaining budget. The cost log is an estimate based on quoted rate; the provider bill is authoritative. GPU time includes CPU simulation while the Pod is running.

## 4. Independent evaluation and verified backup

After checkpoint hashes and the primary 30-seed evaluation pass, set a new `EVAL_OUTPUT` and run `bash scripts/cloud_evaluate.sh`. It evaluates fresh seeds 104000–104009 under nominal conditions, a 5cm fixed-camera shift and a 3cm object reset-position shift, using the same random seed for paired diffusion sampling. Store failures and per-episode traces. This is a separate exploratory OOD check, not model selection.

Set `BACKUP_DIR` to an approved persistent mount and run `bash scripts/cloud_backup.sh`. It copies the run with `rsync --checksum`, builds SHA256 checksums and verifies every destination file. Sync the verified copy back by approved SSH transfer, verify again locally, then stop the GPU Pod under the provider's billing rules. Retained network or volume storage can continue charging after the Pod stops; delete it only after the user confirms backups and retention needs.

## 5. Official benchmark gate

Run official LIBERO only after a pinned Linux LeRobot checkout, dataset revision, checkpoint revision/weights hash, camera keys, 8D/7D schema, action mode and reset behavior are verified. `deploy/cloud/setup_lerobot_linux.sh` creates a separate official environment at a pinned commit; `scripts/libero_official_linux.sh` uses that environment's `lerobot-eval` and fails closed on missing metadata. The official dependency resolution is recorded by the setup script and must be archived for reproduction. FetchPush visual checkpoints are not LIBERO-compatible. SmolVLA needs its own hardware probe and cost approval.
