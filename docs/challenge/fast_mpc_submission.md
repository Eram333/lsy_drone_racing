# Faster MPC leaderboard submission

The candidate configuration selects `level2_mpc_guarded.py`, which inherits the
accepted racing MPC at `TIME_SCALE = 1.4`. It adjusts reference progress only
when observed gate alignment is poor. The original `level2_mpc.py` retains its
1.8 setting and default behavior. No RL or alternative-dynamics changes are included.

The unchanged official 20-race evaluator on 2 November 2026 completed **16/20
races (80%)**, averaging **6.65 seconds** over successful races. This is a fresh
randomized batch; the older 6.50-second result is not a guaranteed lap time.
The existing central personal best before submission was 8.47765 seconds at 85%.
Central validation and the leaderboard update are separate from the local pass.

The central evaluation of `a798cdd` subsequently finished only **9/20 races
(45%)**, below the 50% requirement; the leaderboard was not updated. Setup,
anti-cheat and whitelist checks passed. Reported failures clustered around gate
frames, so the faster timing alone is not a qualified competition result.

The reference governor estimates lateral/vertical alignment time using public
position, velocity and observed gate geometry. A 12 cm approach allowance accounts
for possible frame contact before the centre crossing. Poor alignment reduces the
internal reference rate, bounded between 0.6 and 1.0, with gradual recovery. This
is a ballistic heuristic, not a collision-avoidance guarantee.

Prediction offsets, velocity/acceleration references and crossing deadlines use
the current reference rate. Only the internal trajectory clock changes: the
environment frequency, elapsed simulation time, race step count and success
criterion are unchanged. The original dynamics, input limits and 20 x 40 ms MPC
grid remain in use. The base controller's deadline calculation is extracted into
a method with identical default behavior. All 18 focused MPC tests pass.

The fork's Candidate Qualification workflow runs the unchanged 20-race evaluator
on Ubuntu using the locked Pixi environment. It additionally checks a stronger
development target of 15/20 finishes below 7 seconds.

On 3 November, candidate `a51f3fc` completed **11/20 races (55%)**, averaging
**6.70727 seconds**. It passed the unchanged official evaluator's 50% minimum,
but failed the additional 75% development target. This is a borderline eligible
candidate, not a robust qualification claim. It is submitted in accordance with
the request to publish faster eligible results; central evaluation remains an
independent 20-race run. No success threshold or protected evaluator was changed.

The current central anti-cheat checker passes against upstream main. The protected
simulation, evaluator and dispatch workflow match upstream; only the permitted
controller selection changes in the configuration. All 12 MPC/trajectory unit
tests, full repository Ruff lint, and formatting checks pass. Physics, sensors,
track, frequency, dependencies and lockfile are unchanged by this submission.

Run `pixi run evaluate` to evaluate the selected faster controller. Select
`level2_mpc.py` in `config/level2.toml` to compare the original timing.
