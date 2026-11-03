# V1.2 scientific integrity audit (2026-11-03)

Status: **CPU_VERIFIED** by `python -m worldpolicy.v13_audit`. Machine-readable findings: `results/v13/v12_audit.json`; the auditable per-initial-state ACT/Diffusion table is `results/v13/v12b_paired_initial_states.csv`. Historical V1–V1.2 artifacts were read without modification.

| Question | Finding |
| --- | --- |
| Independent training seeds | Seeds 21, 22 and 23 have different demonstration manifest hashes. These are three distinct training datasets and initializations. |
| Shared training data within seed | ACT and Diffusion in each seed directory refer to the same demonstration manifest. |
| Evaluation units | Each seed uses the same initial-state seeds 60000–60049. There are 150 policy-by-training-seed rows but only **50 unique evaluation initial states**, plus three training realizations. Treating 150 as independent scenes would understate uncertainty. |
| Confirmatory freeze | V1.2b protocol was committed at `f3f82d0774a5114568d06747a70e92de67890746`; V1.2a source was `9252b6c1a68399dc0951920af40fa82c1e9f7bbb`. The frozen protocol and timestamps are stronger evidence than narrative claims; this audit cannot prove no human inspected other exploratory runs beforehand. |
| Pairing | The CSV joins ACT and Diffusion by training seed and initial-state seed. Exact McNemar discordant counts are 22/0, 29/0 and 25/3, respectively. P values were recomputed from episode traces, not aggregate percentages. |
| CPU budget | V1.2b ACT training took 64.1, 74.6 and 69.3 CPU seconds; Diffusion took 4.5, 4.5 and 4.3 seconds. Update counts alone do not match effective training budgets. V1.2c approximately equal CPU control still yielded ACT 22/50 versus Diffusion 8/50 on its separate confirmation set. |
| Friction OOD ID baseline | Its ID seeds are 70000–70049, whereas the main confirmation uses 60000–60049. Rates 23/29/32 versus 27/35/33 therefore describe different initial states. |
| IQL | V1.1 corrected IQL was rerun (3/8 in its own small evaluation). There is no trained FetchPush IQL baseline; the earlier IQL result must not be transferred to FetchPush. |
| CEM | Physics-oracle CEM snapshots MuJoCo before simulating each sampled action sequence and restores the snapshot before real action execution. The oracle may simulate counterfactual futures, but has no access to the actual next state before acting. Learned dynamics receives the current 28D state and candidate actions; the evaluation call does not pass future ground truth. |

The three-seed V1.2b confirmation rates remain ACT 27/50, 35/50, 33/50 and Diffusion 5/50, 6/50, 11/50. These are **privileged state-input FetchPush simulation** results, not RGB or VLA results. The effect is robust within this narrow implementation and protocol, but the much smaller Diffusion compute budget and compact MLP denoiser prevent an algorithm-level claim. The V1.2c equal-CPU control narrows that concern without removing differences in architecture, search effort or convergence.

Reproduce the audit from the repository root:

```powershell
.\.venv\Scripts\python.exe -m worldpolicy.v13_audit
```

The CSV is retained locally under `results/v13/`; it is ignored by Git along with the original large result artifacts. The audit code is committed.
