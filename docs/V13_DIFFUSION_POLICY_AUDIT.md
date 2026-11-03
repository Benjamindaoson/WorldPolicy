# Diffusion Policy algorithm audit

Status: **TESTED** for code/formula inspection; V1.2/visual training is **CPU_VERIFIED**. This repository's `DiffusionPolicy` is a compact conditional DDPM baseline, **not** the official Stanford image policy. The [Stanford reference implementation](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py) uses a `ConditionalUnet1D`, image encoder, action normalizer and configurable observation, prediction and execution horizons.

| Component | Local implementation | Consequence |
| --- | --- | --- |
| Forward schedule | 20-step cosine cumulative alpha, beta clipped to `[1e-4, .999]` | Nonzero noise at every step; no claim of optimal schedule. |
| Prediction target/loss | Epsilon prediction, mean squared error | Matches a standard DDPM objective and reference's epsilon option. |
| Reverse scheduler | Ancestral DDPM posterior variance, all 20 steps, final action clipping | Formula unit tests cover finite/bounded sampling, but no sampler search. |
| Action scale | Environment actions already bounded `[-1,1]`; no dataset-fitted action normalizer | Appropriate for this FetchPush command contract, but unlike the reference pipeline and possibly suboptimal per-axis. |
| Denoiser | Flattened action-chunk MLP, 128 hidden | Much less temporal inductive bias/capacity than reference 1D U-Net or Transformer. |
| Horizons | State-only: current 28D observation, 4 predicted actions, execute first action. Visual: two RGB/proprio observations, 4 predicted actions, execute first action. | Both use closed-loop one-step replanning; visual history is short. |
| Convergence/budget | V1.2b Diffusion ~4.3–4.5 CPU seconds versus ACT ~64–75 seconds. V1.2c equal-CPU control exists. | V1.2b is not compute matched. V1.2c still has no architecture or hyperparameter parity. |

No evidence supports the general statement that Diffusion Policy is weaker than ACT. The observed gap applies to **these trained checkpoints**. A stronger test would freeze train/validation/test episode seeds, search denoiser family/capacity, normalization, horizon, sampling steps, learning rate and training budget on validation only, then evaluate the selected settings on untouched test episodes across at least three training seeds. Preserve failed configurations and report actual CPU/GPU time. The V1.3 visual run is a first implementation check and does not satisfy that full search protocol.

The local model's [source](../src/worldpolicy/policies.py) and [visual wrapper](../src/worldpolicy/visual_policy.py) specify the exact equations and tensor contracts. The [reference Transformer denoiser](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/transformer_for_diffusion.py) is a second unimplemented architecture option.
