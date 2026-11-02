# Faster MPC leaderboard submission

The level-2 configuration selects `level2_mpc_fast.py`, which inherits the accepted
racing MPC and uses the previously explored `TIME_SCALE = 1.4`. The original
`level2_mpc.py` retains its 1.8 setting for comparison. No experimental SMPC or RL
changes are included.

The unchanged official 20-race evaluator on 2 November 2026 completed **16/20
races (80%)**, averaging **6.65 seconds** over successful races. This is a fresh
randomized batch; the older 6.50-second result is not a guaranteed lap time.
The existing central personal best before submission was 8.47765 seconds at 85%.
Central validation and the leaderboard update are separate from the local pass.

The current central anti-cheat checker passes against upstream main. The protected
simulation, evaluator and dispatch workflow match upstream; only the permitted
controller selection changes in the configuration. All 12 MPC/trajectory unit
tests, full repository Ruff lint, and formatting checks pass. Physics, sensors,
track, frequency, dependencies and lockfile are unchanged by this submission.

Run `pixi run evaluate` to evaluate the selected faster controller. Select
`level2_mpc.py` in `config/level2.toml` to compare the original timing.
