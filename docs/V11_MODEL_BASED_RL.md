# Model-based RL status

Status: NOT_RUN. The current dynamics ensemble plus CEM is a planner, not model-based policy optimization. No imagined replay, reward/termination model, actor-critic update from model rollouts, or real-vs-imagined return gap exists yet. Continuous IQL performs real offline parameter updates on point-reach transitions, but this is model-free offline RL. CQL and policy distillation are also NOT_RUN.

Required next gate: train and validate MuJoCo arm reward/termination dynamics on held-out transitions, calibrate uncertainty, cap imagination horizon, update an actor and critic using imagined and real replay, and evaluate exclusively in held-out MuJoCo physical rollouts. Track equal real-interaction and compute budgets against model-free controls.
