# Experiment protocol

Status: IMPLEMENTED for the small MuJoCo reach run; formal benchmark protocol NOT_RUN.

The code records dataset episode IDs, manifest hashes, train seed, optimizer steps, checkpoint hashes, simulator version, evaluation seeds, per-step trajectories and success. ACT and Diffusion share the same MuJoCo training dataset and paired evaluation seeds. A Wilson 95% interval is reported for binomial success; it does not account for training-seed variation. The point-reach CEM diagnostic shares test initializations but varies planner optimization settings and is exploratory.

For confirmatory studies: freeze the protocol and code revision before collecting new test seeds; use at least three independent training seeds, paired task/initial-state evaluation, and independent scene/task groups. Report all failures, per-task success, worst group, compute and interaction budgets. Never reuse diagnostic seeds as a confirmatory test after selecting a horizon on them.
