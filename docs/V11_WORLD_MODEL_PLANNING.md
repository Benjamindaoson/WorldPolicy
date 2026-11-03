# World-model planning contract

Status: point-reach physics oracle and learned model CPU_VERIFIED; MuJoCo learned model NOT_RUN; WorldDynamics arm adapter BLOCKED on protocol mismatch.

`ActionSpec` requires embodiment, dimension, unit, frame, control frequency and bounds to match exactly. `WorldModelMetadata` also requires checkpoint ID, version, horizon and evidence type. CEM checks this before planning and checks predicted tensor shapes. The point-reach learned ensemble consumes 2D world-frame m/s actions at 10 Hz. The new MuJoCo arm consumes 2D joint rad/s actions at 20 Hz. They must not be connected. Read-only inspection found WorldDynamics' current MuJoCo data scene to be a free-joint cube, not this arm. No WorldDynamics weights were imported.

To enable arm planning, train a new dynamics model on arm transitions, export explicit state/action mappings and uncertainty calibration, then run paired base-policy/oracle/learned planning on fresh test seeds. An interface match alone is not evidence of useful planning.
