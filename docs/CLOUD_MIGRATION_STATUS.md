# Cloud migration status

| Item | Status | Evidence / next gate |
| --- | --- | --- |
| Branch and local Git history | TESTED | `develop/worldpolicy-v1.3`; Git bundle generator and bundle verification script; no remote push. |
| Isolated Linux Python environment | IMPLEMENTED | `deploy/cloud/setup_linux.sh`, 52-package hash-pinned Linux lock for Python 3.12; not yet executed on Linux. |
| CUDA diagnostics and short training probe | CPU_VERIFIED locally / GPU_SIZING_REQUIRED | `scripts/cloud_preflight.py`, real CPU five-step record; CUDA probe awaits authorized Linux instance. |
| Visual training and resume | CPU_VERIFIED smoke / GPU NOT_RUN | `visual_experiment --device cuda --resume`, 100-update progress checkpoints. |
| Official LIBERO and SmolVLA | BLOCKED | Linux stack and pinned official checkpoint/dataset not yet available. |
| Data and checkpoint integrity | TESTED locally | SHA256 dataset manifests, checkpoint hashes, `scripts/cloud_verify.py`; cloud transfer not performed. |
| GPU/price monitor and spend ceiling | IMPLEMENTED | `cloud_train.sh` logs `nvidia-smi`, estimated accrued cost and uses `timeout`; no paid run. |
| Verified backup | IMPLEMENTED, NOT_RUN on cloud | `cloud_backup.sh` uses rsync plus full SHA256 verification on an approved persistent destination. |
| Multi-GPU | NOT_RUN | Not needed for compact FetchPush visual run; future scaling needs NCCL/topology probe. |

Current GPU: local GTX 1650 4GB with a CPU-only PyTorch environment. Next recommended resource: a provisional 1× RTX 4090 24GB Linux Pod. Exact minimum VRAM and mixed precision status await the cloud probe. The selected first run is estimated at 1–2 GPU hours and $0.89–$1.78 GPU charge at the [Runpod Secure Cloud quote](https://www.runpod.io/pricing), plus persistent storage. No paid cloud instance has been launched or authorized.
