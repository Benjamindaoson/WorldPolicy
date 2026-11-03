# GPU resource requirements — 2026-11-03

## Measured local gate

Status: **CLOUD_GPU_REQUIRED** for official Linux LIBERO and the requested formal cloud visual training. **GPU_SIZING_REQUIRED** for the exact VRAM floor: the local PyTorch build is CPU-only, so no CUDA peak has been measured. `nvidia-smi` reports a GeForce GTX 1650 with 4096 MiB, driver 457.49, CUDA driver API 11.1, and 1313 MiB used by the Windows desktop at inspection. `torch 2.15.0+cpu` reports `cuda.is_available() == False`.

`scripts/cloud_preflight.py` ran five *actual optimizer updates* per model on real `task_visible` FetchPush chunks (8 smoke episodes, 188 chunks), batch 32, float32. Machine-readable record: `artifacts/v13/cloud_preflight_cpu.json`.

| Policy | Parameters | Median CPU step, first excluded | 1500-step CPU projection | CUDA peak |
| --- | ---: | ---: | ---: | --- |
| Vision-only ACT | 241,348 | 0.129 s | 193 s | Unmeasured |
| Vision + proprio ACT | 250,884 | 0.124 s | 186 s | Unmeasured |
| Vision + proprio DDPM | 108,080 | 0.083 s | 124 s | Unmeasured |

The projected 503 seconds is **optimizer time only**. It excludes generating 128 train and 24 validation episodes, 30 independent test episodes per policy, CPU MuJoCo rendering, checkpointing, transfer and startup. The target contract is two 64×64 RGB views, history 2, four-action prediction, batch 32, 1500 updates *per policy*. Float16/bfloat16 compatibility is unverified; the current trainer is float32. No distributed training or NCCL is needed for these compact models. The local CPU can run smoke tests; a numerical claim that this model needs 24GB would be unsupported.

## Provisional first rental

Choose **one Linux RTX 4090 24GB Pod**, preferably Runpod Secure Cloud, as a practical unified target for visual policies and the later official LIBERO stack. This is a *provisional choice*, not a measured minimum. On 2026-11-03, [Runpod's pricing page](https://www.runpod.io/pricing) lists RTX 4090 Secure Cloud at **$0.89/GPU-hour** (Community Cloud $0.34/hour), RTX 5090 at $1.19/hour, and standard persistent network storage below 1TB at $0.07/GB/month. Availability and checkout quote must be checked before authorization.

Plan **1–2 billed GPU hours** for environment setup, 5-step GPU probe, one 3-policy 1500-update visual run, independent evaluation and backup: **$0.89–$1.78 GPU charge** at the quoted Secure Cloud rate. This is a planning envelope, *not* a measured GPU runtime. A 20GB persistent network volume would add **$1.40/month** if retained for the full month; a stopped Pod's attached volume may keep billing. The first cloud probe must report `peak_allocated_vram_bytes`, `peak_reserved_vram_bytes`, step time, GPU utilization and GPU identity. Then set a defensible numeric minimum (measured peak plus headroom), revise the hours and authorize a maximum charge before formal training. Separate LIBERO/SmolVLA fine-tuning estimates require their own real probe and dataset/checkpoint revisions.

Do not launch a paid Pod, buy storage, or increase a cap from these estimates alone. `scripts/cloud_train.sh` enforces a positive current hourly quote, a user-approved maximum, a measured GPU probe, an exact dataset hash and a runtime timeout. It does not create an instance.
