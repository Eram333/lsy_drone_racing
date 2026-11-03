# CI compatibility repair

The accepted `level2_sprint.py` policy, all its dependency modules and weight
archives, and the selected level-2 configuration are unchanged. SHA-256 checks
confirm all 30 recorded policy/configuration files are byte-identical to the
accepted `ee59ff5` checkout. This repair does not claim a new race result.

The latest testing job stopped during collection because legacy MPC exports and
planner modules were removed while their tests remained. Restore the matched
`level2_mpc.py`, `attitude_mpc.py`, `racing_trajectory.py` and
`trajectory_gradient.py` implementations from the last green `d2f8b6e` revision.
The active sprint policy does not import these legacy MPC modules. Their backup
MPC interface is restored instead of deleting or weakening the tests.

Fix the remaining import sorting and benchmark formatting. Protected simulation,
evaluation and competition-dispatch scripts, physics and scoring are unchanged.

Validation: all **32 targeted MPC/trajectory tests pass**, repository Ruff lint
passes, and all **100 Python files** pass formatting. Full Linux test-suite
validation is left to the existing Testing workflow; no full-suite pass is
claimed from the local Windows run.
