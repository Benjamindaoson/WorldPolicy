"""Read-only audit of saved V1.2 evidence and paired outcomes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .statistics import exact_mcnemar_pvalue, paired_counts


def run(root: Path, output: Path) -> dict:
    a = json.loads((root / "fetch_push" / "summary.json").read_text())
    b = json.loads((root / "fetch_push_b" / "summary.json").read_text())
    ood = json.loads((root / "fetch_push_ood.json").read_text())
    paired = []
    summary = {"status": "CPU_VERIFIED", "v12a_source_commit": a["source_commit"],
               "v12b_source_commit": b["source_commit"], "training_seeds": [],
               "same_initial_states_across_training_seeds": True,
               "confirmatory_seed_start": b["confirmatory_seeds"][0],
               "ood_seed_start": ood["evaluation_seeds"][0]}
    for seed in (21, 22, 23):
        name = str(seed)
        left = b["training_runs"][name]
        previous = a["training_runs"][name]
        assert previous["dataset_manifest_sha256"] == left["dataset_manifest_sha256"]
        run_dir = root / "fetch_push_b" / f"seed_{seed}"
        act = json.loads((run_dir / "act" / "evaluation.json").read_text())
        diffusion = json.loads((run_dir / "diffusion" / "evaluation.json").read_text())
        a_rows, d_rows = act["confirmatory_traces"], diffusion["confirmatory_traces"]
        assert [r["seed"] for r in a_rows] == [r["seed"] for r in d_rows] == b["confirmatory_seeds"]
        assert act["dataset_manifest_sha256"] == diffusion["dataset_manifest_sha256"]
        counts = paired_counts([r["success"] for r in a_rows], [r["success"] for r in d_rows])
        assert counts == left["paired_act_diffusion"]
        for ar, dr in zip(a_rows, d_rows, strict=True):
            paired.append({"training_seed": seed, "evaluation_seed": ar["seed"],
                           "act_success": int(ar["success"]),
                           "diffusion_success": int(dr["success"]),
                           "act_final_distance_m": ar["goal_distance"],
                           "diffusion_final_distance_m": dr["goal_distance"]})
        summary["training_seeds"].append({"seed": seed,
            "dataset_manifest_sha256": left["dataset_manifest_sha256"],
            "act_checkpoint_sha256": act["training"]["checkpoint_sha256"],
            "diffusion_checkpoint_sha256": diffusion["training"]["checkpoint_sha256"],
            "paired_counts": counts, "exact_mcnemar_p": exact_mcnemar_pvalue(counts),
            "act_train_seconds": act["training"]["seconds"],
            "diffusion_train_seconds": diffusion["training"]["seconds"]})
    output.mkdir(parents=True, exist_ok=True)
    with (output / "v12b_paired_initial_states.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    summary["paired_rows"] = len(paired)
    summary["independent_eval_initial_states"] = len(b["confirmatory_seeds"])
    summary["ood_id_baseline_differs_reason"] = "OOD uses seeds 70000-70049; primary confirmation uses 60000-60049"
    (output / "v12_audit.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/v12"))
    parser.add_argument("--output", type=Path, default=Path("results/v13"))
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.output), indent=2))


if __name__ == "__main__":
    main()
