# Closed-loop FetchPush planning

Status: CPU_VERIFIED exploratory simulation, seeds 30000–30009. The physics oracle uses a MuJoCo state snapshot, simulates only sampled future actions, and restores the exact current state before executing the selected action. No real future observation enters action selection. Random shooting and CEM each evaluate 32 three-step candidate sequences per control action. The learned ensemble uses the same CEM population, iterations, horizon, bounds and terminal cost.

| Policy/planner | Successes / 10 |
| --- | ---: |
| Geometric demonstration controller | 9 |
| Physics oracle random shooting | 10 |
| Physics oracle CEM | 10 |
| Learned robot dynamics CEM | 3 |
| Learned dynamics CEM, uncertainty weight 10 | 2 |

The first seed was already within the 0.05 m success threshold; all methods stopped in one step there. This is a tiny exploratory set and the oracle's 10/10 does not establish benchmark performance. The learned model's one-step prediction error was small, but candidate ranking and contact prediction errors compound in closed loop. The large uncertainty penalty did not help. Raw action/reward trajectories: `artifacts/v12/fetch_push_oracle_10.json` and `fetch_push_learned_10.json`; prediction diagnostics: `artifacts/v12/fetch_push_dynamics/diagnostics.json`. The learned planner **underperformed** the oracle and the trained ACT baseline on their separate evaluation protocols; cross-protocol success rates must not be compared causally.
