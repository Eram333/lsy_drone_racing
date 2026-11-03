# Adaptive final-leg timing - 3 November 2026

## Decision

**Keep the adaptive final-leg timing.** Across 20 paired seeds it reduces mean
successful race time from **7.875 to 7.715 s (2.03%)**, with the same **16/20
finishes and four contacts**, and no newly failed baseline-success seed.
The 3-4 s target is not achieved, and baseline reliability problems remain.
The finish-time default remains `Level2MPC.FINISH_TIME_FACTOR = 0.75`; setting
it to `1.0` disables this extra retiming. A later aggressive-timing experiment
selected global scale 1.45 with analytic initialization; see
[the follow-up results](./mpc_optimization.md).

## Scope and investigation

This follow-up keeps `TIME_SCALE = 1.7`, the original nominal optimizer, MPC
weights, 20-stage/0.8 s horizon, actuator bounds, and frame/pole clearances.
Simulation physics, track randomizations, race rules and scoring are unchanged.

The earlier faster profile changed both the nominal optimizer and global timing;
its failures cannot be attributed to timing alone. The saved Gate 4 approach
traces show why a simple force-limit explanation is insufficient:

| Seed | Original / earlier faster approach tracking RMS | Original / faster tilt-command saturation fraction |
| --- | ---: | ---: |
| 203 | 9.28 / 9.04 cm | 10.1 / 13.7% |
| 204 | 7.64 / 6.39 cm | 4.3 / 0.0% |
| 208 | 6.93 / 8.09 cm | 7.2 / 7.7% |

These means cover the recorded approach intervals; failed approaches are
truncated, so this is not a matched-duration error comparison.

The earlier faster profile collided on these approaches despite modest average
errors and, for seed 204, no command saturation. Late lateral alignment, vehicle
footprint near the bars, and model/repair transients need attention, not just
larger gains or smaller clearances. Zero clipping does not rule out other forms
of actuator or model limitation.

## Isolated change

The existing planner now considers shorter **final-leg** timing only after the
penultimate passage is confirmed, at the existing reference boundary. Therefore
the whole approach to Gate 4 remains the original controller's trajectory. A late
reference phase retains the original finish rather than splicing mid-polynomial.

Candidate duration factors 0.75, 0.80, ..., 0.95 are checked from fastest to
slowest. The last crossing's velocity is adjusted; the preceding shared position,
velocity and acceleration are unchanged. Accepted quintics retain the C2 join.
This introduces no additional position/velocity/acceleration jump at that join;
it does not fix pre-existing sensor-repair or reference-clock discontinuities.

Each candidate is screened at 81 points for frame/pole/floor clearance, speed,
the existing force envelope, calibrated collective thrust, and the identified
rotational model's required commands. The latter include desired angular
acceleration, angle and rate terms. Nominal headroom is reserved within the
existing limits: 90% of attitude-command bounds and 95% of collective maximum.
If no candidate passes, the plan is unchanged. These sampled, nominal-model
checks are not a robust flight certificate or a full individual-motor model.

Implementation: [controller](../../lsy_drone_racing/control/level2_mpc.py),
[planner](../../lsy_drone_racing/control/racing_trajectory.py),
[focused tests](../../tests/unit/test_finish_retiming.py).

## Development comparison

The original saved measurements were reused after checking all four core source
hashes and exact replay prefixes; no original development races were rerun.
Screening used seed 0 and the previously difficult seeds 203, 204 and 208. The
remaining development seeds completed the existing 200-209 comparison.

| Seeds 200-209 | Mean successful race | Finishes | Contacts |
| --- | ---: | ---: | ---: |
| Original | 7.8667 s | 9/10 | 1 |
| Adaptive finish | 7.6822 s | 9/10 | 1 |

The paired gain is **2.34%**, with no new failed seed. The failure at seed 201 is
unchanged at 3.88 s, before activation. Final-leg factors range from 0.75 to 1.0;
seed 205 retains its complete original plan. Across all 11 development/screening
races, every recorded state, reference and action through Gate 4 matches the
saved original bit-for-bit (or matches through an earlier original failure).
This is an intentionally small finish improvement,
not a claim that the Gate 4 braking problem has been solved.

Development data: [screening](../../.pixi/adr_followup_20261103/finish_screen.json),
[remaining seeds](../../.pixi/adr_followup_20261103/finish_development.json),
[original](../../.pixi/adr_20261102/baseline_validation.json).

## Untouched holdout validation

Seeds **300-309** were reserved before validation. Candidate code and limits
were frozen; no holdout result was used for tuning. Only the original and the
same candidate were run, sequentially on the existing Windows CPU environment.

| Fresh seeds 300-309 | Mean successful race | Finishes | Contacts |
| --- | ---: | ---: | ---: |
| Original | 7.8857 s | 7/10 | 3 |
| Adaptive finish | 7.7571 s | 7/10 | 3 |

The gain is **1.63%** over the identical seven successful seeds. The three failures
(301, 303, 304) are reproduced at the same times, before activation. No candidate
finish is slower than its matching original. On every fresh seed the complete
recorded state/reference/action prefix through Gate 4, including new target-frame
and quaternion telemetry, is bit-for-bit identical to original. No uncredited
target-plane crossing or boundary failure was observed.

This fresh baseline has 70% success, not the 90% measured on 200-209. Equal paired
outcomes are evidence for this increment, not proof of a population-level
reliability guarantee. Average per-race positional RMS is essentially unchanged
(5.706 cm original, 5.725 cm candidate on the fresh set).

Raw results: [fresh original](../../.pixi/adr_followup_20261103/baseline_holdout.json),
[fresh candidate](../../.pixi/adr_followup_20261103/finish_holdout.json).

## Runtime and checks

The sequential fresh batches recorded control p95 **18.44 / 28.69 ms** and MPC
solve p95 **12.68 / 19.27 ms** (original / candidate). These adverse wall-clock
measurements are retained, not hidden. To investigate load/order effects without
repeating a batch, one extra seed-300 pair was run in the **same process**, this
time candidate first and original second. Mean MPC solve time was **9.07 ms for
both**; control p95 was **24.76 ms candidate / 26.22 ms original**. The new
feasibility search itself took **6.30 ms once**, at the final-leg transition.

Thus there is measurable one-time overhead and variable wall-clock timing; no
real-time speed improvement or guaranteed 20 ms deadline is claimed. The timing
diagnosis does not alter the race-time comparison or failure counts.
Replay data: [candidate](../../.pixi/adr_followup_20261103/runtime_candidate.json),
[original](../../.pixi/adr_followup_20261103/runtime_original.json).

All **26 focused tests pass**, including inverse rotational-input verification
against the actual CasADi dynamics, feasibility rejection, reverse crossing,
unchanged prefixes and C2 joins. Ruff, editor diagnostics and `git diff --check`
were checked. Existing uncommitted work was preserved; the original nominal
optimizer, trajectory-gradient module, attitude MPC and prior notes were not
replaced. No full test suite or repeated full randomized evaluation was run.

## Reproduction and remaining bottlenecks

[The benchmark](../../benchmarks/racing_mpc.py) still wraps the existing project
simulation loop, with explicit identical reset seeds and headless rendering.
Telemetry now also saves observed target frames, vehicle quaternions/angular
velocities, and final trajectory arrays for offline diagnosis. No private
simulation state is supplied to the controller.

Saved original sources are in `.pixi/adr_followup_20261103/original`. Use the
existing detached Windows CPU Pixi interpreter documented in the
[previous report](mpc_optimization_20261103.md), for example:

```powershell
& $adrPython benchmarks/racing_mpc.py --seeds 300,301,302,303,304,305,306,307,308,309 --output .pixi/adr_followup_20261103/replay.json
```

Add `--source-dir .pixi/adr_followup_20261103/original` for the original controller.

The original successful development races spend **6.342 s on average before the
fourth crossing**, so final-leg timing alone cannot reach 3-4 s. Next work should
address earlier lateral braking and body-aware clearance throughout gate entry
and exit, then continuous, actuator-feasible sensor-triggered repairs. The new
holdouts also expose earlier gate contacts; globally compressing the schedule
would amplify those unresolved problems. No controller rewrite or MPCC switch
was made, and no commit or push was performed.
