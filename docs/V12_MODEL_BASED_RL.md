# Model-based RL status

Status: NOT_RUN. Robot-conditioned dynamics and CEM are implemented and tested, but no actor/critic has been updated with imagined FetchPush rollouts. There is no learned reward/termination head, uncertainty-gated imagined replay, or equal-interaction model-free control. The learned CEM planner's 3/10 exploratory result also fails the current planning reliability gate. Calling it model-based RL would be incorrect.

Next gate: improve contact-state model calibration on validation-only data, add reward/termination prediction, train an actor/critic with bounded imagination horizon and real replay, then evaluate saved policies on fresh independent MuJoCo seeds with matched real-interaction budgets. CQL, corrective learning and distillation remain NOT_RUN.
