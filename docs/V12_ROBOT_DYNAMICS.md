# FetchPush robot-conditioned dynamics

Status: CPU_VERIFIED. The point-reach and WorldDynamics cube models are not used to predict FetchPush robot actions. `src/worldpolicy/fetch_push_dynamics.py` trains a three-member 28D state / 4D action residual MLP ensemble using actual MuJoCo FetchPush transitions. The action contract specifies normalized Cartesian delta at 25 Hz. The model holds the exogenous goal fixed during predictions and returns ensemble variance as an uncertainty proxy.

Training data: 64 episodes mixing geometric expert and uniform random actions, 3,107 transitions; 1,500 updates per member. An independent eight-episode validation set yielded one-step full-state MSE `0.00018985`. Checkpoint SHA256: `7e871ea500b41fb9810812b13399f9253a919425a63b6c046ee4d9b15f859a0c`. Raw transition arrays, checkpoint and metrics are in `artifacts/v12/fetch_push_dynamics/`.

On ten exploratory initial states, random candidate three-step state MSE grew from `1.59e-5` at one step to `4.37e-5` at three steps. Mean three-step object XY MSE was `1.10e-5`; mean absolute selected-plan cost error was `0.00446`. Candidate rank correlation per initial state ranged roughly 0.80–0.95. These aggregate errors did not translate to reliable closed-loop planning; see the separate planning report. The model does not predict reward/termination and is not a model-based RL system.
