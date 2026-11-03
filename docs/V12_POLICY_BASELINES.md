# FetchPush policy baselines

Status: CPU_VERIFIED, simulation only. Official `FetchPushDense-v4`, state-only 28D input, 4D normalized Cartesian/gripper action, 25 Hz, maximum 80 steps. No RGB or language input enters these local ACT/DDPM policies. Dataset, effective batch, action chunk and evaluation initial states are paired within each training seed.

## Frozen V1.2a, 500 updates, 50 confirmatory states 40000–40049

| Training seed | ACT | Diffusion | Expert demonstrations / 64 |
| --- | ---: | ---: | ---: |
| 21 | 2/50 | 2/50 | 55 |
| 22 | 2/50 | 2/50 | 58 |
| 23 | 2/50 | 5/50 | 59 |

The ACT zero-latent action collapse and unstandardized state inputs motivated a new protocol. These are valid negative outcomes under their frozen settings, not erased failures. Raw records: `artifacts/v12/fetch_push/`.

## Frozen V1.2b, 2000 updates, fresh 50 confirmatory states 60000–60049

| Training seed | ACT | ACT Wilson 95% | Diffusion | Diffusion Wilson 95% | Exact paired McNemar p |
| --- | ---: | --- | ---: | --- | ---: |
| 21 | 27/50 | [0.404, 0.670] | 5/50 | [0.043, 0.214] | 4.77e-7 |
| 22 | 35/50 | [0.562, 0.809] | 6/50 | [0.056, 0.238] | 3.73e-9 |
| 23 | 33/50 | [0.522, 0.776] | 11/50 | [0.128, 0.352] | 2.74e-5 |

These p-values compare paired success outcomes **within** each training seed. The same 50 initial states are reused across training seeds; do not pool them as 150 independent states. ACT took 64–75 s CPU training per seed versus Diffusion 4–5 s, despite matching update counts. Median inference latencies were about 1.8–2.1 ms ACT versus 6.3–7.4 ms Diffusion. Raw traces/checkpoints: `artifacts/v12/fetch_push_b/`. One success and one failure per policy from seed 21 were replayed against their exact checkpoint/actions and encoded as local MP4 files.

## Frozen V1.2c, approximately equal CPU training time, fresh states 80000–80049

Training seed 21 and the same demonstration dataset: ACT 2000 updates in 64.1 s, Diffusion 30,000 updates in 61.5 s. ACT succeeded 22/50 (Wilson [0.312, 0.577]); Diffusion 8/50 (Wilson [0.083, 0.285]). Paired counts: both success 6, ACT-only 16, Diffusion-only 2, both fail 26; exact McNemar p=0.00131. This single-seed control is suggestive for this task and architecture, not a general algorithm ranking. Raw records: `artifacts/v12/fetch_push_compute_control/`.

## Object-friction OOD, paired seeds 70000–70049

The same V1.2b checkpoints were evaluated twice per seed: nominal contact friction and doubled object friction. This is a physics shift, not unseen object appearance or LIBERO OOD.

| Training seed | ACT nominal → OOD | Diffusion nominal → OOD |
| --- | ---: | ---: |
| 21 | 23/50 → 16/50 | 2/50 → 4/50 |
| 22 | 29/50 → 15/50 | 3/50 → 2/50 |
| 23 | 32/50 → 12/50 | 12/50 → 9/50 |

ACT degrades under increased object friction in all three training seeds. The Diffusion rates are low in both conditions, and small changes should not be read as robustness improvement. Per-initial-state outcomes and paired counts are in `artifacts/v12/fetch_push_ood.json`. The OOD seed range is separate from all confirmation ranges; nominal and shifted episodes use identical initial seeds.
