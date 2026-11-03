# MPC optimization notes — 23 October 2026

The three-second target has not been achieved. A planned trajectory duration is
not an achieved lap time, and passing lint is not passing the competition flight
evaluation.

## Local speed tuning — 24 October 2026

The default `Level2MPC.TIME_SCALE` is now **1.7**, previously 1.8. The existing
trajectory scaling adjusts segment durations, crossing velocities and
accelerations together. Frame/pole clearances, the MPC, track, physics and
competition evaluator are unchanged.

Paired Windows CPU screening used the same level-2 seeds, with rendering disabled
and seeds set in memory rather than editing the race configuration:

| Seed | Previous 1.8: flight time / gates | Selected 1.7: flight time / gates |
| --- | --- | --- |
| 0 | 8.50 s / 5 | 8.08 s / 5 |
| 1 | 8.52 s / 5 | 8.06 s / 5 |
| 4 | 9.60 s / 5 | 9.06 s / 5 |
| 7 | 8.50 s / 5 | 8.06 s / 5 |
| 100 | 8.54 s / 5 | 8.06 s / 5 |
| 101 | Failed at 4.16 s / 2 | Failed at 3.92 s / 2 |

Both settings finished 5/6 races. Mean successful flight time decreased from
8.732 s to 8.264 s (5.36%). These are simulated flight times, not program runtime.
This small tuning set does not establish general reliability or a new central
competition result. More aggressive 1.65 and intermediate 1.75 settings failed
seed 1 and were rejected. Online replanning makes outcomes non-monotonic in the
time scale. The 12 existing planner/MPC unit tests and Ruff check passed.

## Online planning runtime — 24 October 2026

The standard controller now supplies an analytic derivative to the existing
L-BFGS-B online repair. `trajectory_gradient.py` reproduces the original sampled
cost, including signed gate frames, poles, thrust, jerk and position/velocity
continuity. Topology is cached, while each new observation rebinds geometry and
cost parameters. A compact matrix graph generates a native C library under
`.pixi/trajectory_costs` using the existing compiler. Compilation happens before
the control loop and is reused across starts; without a usable compiler, CasADi
evaluates the same graph directly. No extra packages are installed.

The nominal initialization optimizer, time scale, physics, clearance settings
and control frequency are unchanged by this runtime revision. Quintic
coefficients are reused until geometry changes. Nonfinite or higher-cost repair
results are rejected. Initial acados slack costs are explicitly set to their
previous fallback values, removing that warning without changing its cost.

Paired headless CPU timings for the standard time scale 1.7:

| Seed | Flight before / after | Gates before / after | Total repair before / after | Longest control step before / after | Missed 20 ms deadlines before / after |
| --- | --- | --- | --- | --- | --- |
| 0 | 8.08 / 8.08 s | 5 / 5 | 19.652 / 0.429 s | 5.290 / 0.122 s | 51 / 14 |
| 1 | 8.06 / 8.08 s | 5 / 5 | 15.402 / 0.338 s | 3.193 / 0.102 s | 56 / 20 |

Total online repair time decreased about 97.8% across these paired runs. New
verification seeds 4 and 100 also finished all gates, in 9.08 and 8.06 s, with
longest control steps of 80 and 86 ms. All four new runs had a 95th-percentile
control time below 20 ms, but occasional deadline misses remain. These timings
depend on hardware and background load and are not hard real-time guarantees.
They improve program runtime, not the simulated lap to four seconds.

All 18 affected unit tests pass. New tests compare the generated cost and
gradient against the independent NumPy objective and central finite differences,
including sensor updates, reverse crossings, empty pole sets, repair continuity,
and rejection of invalid optimizer output. The initial optimizer can still
report nonconvergence; its reason is now logged as well.

The remaining JAX cast warning was traced to MuJoCo MJX `collision_convex._box_box`:
`jp.where(jp.isinf(dist), jp.finfo(float).max, dist)` casts a float64 maximum
into the configured smaller JAX dtype during initial contact setup. It is not a
controller-generated overflow. That dependency code and the physics precision
have not been changed, and the warning has not been suppressed. The unsupported
CasADi-version notice and local Pixi/SSL setup warnings also remain visible.

## Selective takeoff and fourth-passage timing — 24 October 2026

The selected profile retains the overall time scale 1.7 and shortens only the
first and fourth legs by 5%. The fourth passage's crossing-velocity vector is
increased by 5%, preserving its crossing direction. Factors address passage
order, including repeated physical gates. Quintics are reconstructed using
shared endpoint velocities and accelerations, preserving C2 continuity rather
than introducing reference jumps. The online repair keeps the retimed durations
and normal crossing speeds fixed as before.

Headless paired screening with unchanged level-2 settings:

| Seed | Previous flight / gates | Selected flight / gates |
| --- | --- | --- |
| 0 | 8.08 s / 5 | 7.90 s / 5 |
| 1 | 8.08 s / 5 | 7.90 s / 5 |
| 4 | 9.08 s / 5 | 8.92 s / 5 |
| 100 | 8.06 s / 5 | 7.86 s / 5 |
| 101 | Failed at 3.80 s / 2 | 7.88 s / 5 |
| 7 (additional check) | 8.06 s / 5 | 7.88 s / 5 |
| 42 (additional check) | 8.08 s / 5 | 7.94 s / 5 |

The selected profile completed 7/7 races, compared with 6/7 for the previous
profile. Across the six paired finishes, mean flight time decreased from 8.240 to
8.067 seconds (2.10%). On seeds 0 and 1, reaching 40 cm altitude took 0.90 and
0.96 s, previously 0.94 and 1.00 s. Seed 0's fourth crossing speed increased
from 0.858 to 0.934 m/s; seed 1's average speed on the fourth leg increased from
1.009 to 1.060 m/s, while its instantaneous crossing speed slightly decreased.
These are measured improvements, not a claim that every velocity increases.

Tracking stability is mixed: seed 4's positional RMS error decreased from
8.93 to 7.35 cm and its 95th-percentile angular speed decreased from 6.61 to
5.37 rad/s. Seed 100's positional RMS error increased from 4.32 to 5.68 cm.
A small screening set does not establish general reliability or four-second
flight. The existing height trim and single-stage gate corridor are retained.

Rejected local tests combined 10% retiming with a current-height integral and
finished only 3/4 races. Extending gate constraints over several prediction
stages made that candidate worse. Those variants remain local experiments.
All 20 affected unit tests and Ruff checks pass. Regression tests check physical
endpoint continuity, reverse/repeated passages, input immutability and invalid
timing factors. The selected factors are active in `level2_mpc.py`; the standard
Windows launcher needs no additional controller argument. A final seed-0 run
using the actual default controller reproduced 7.90 s with all five passages.
Occasional control deadline warnings and the previously documented dependency
warnings remain.

## Eight-second MPC reliability revision

The current focus is the existing MPC's reliability. The track, physics,
control frequency, sensor range, competition workflow and evaluator are unchanged.

A reproducible failure on seed 4 hit gate 3 before its centre crossed the plane:
local position was approximately `[-0.075, -0.129, 0.002]` m. The reference itself
ran close to the frame, and approximately 4 cm of additional tracking error
consumed the remaining clearance. Sampled frame clearance was raised from
11 to 15 cm and pole clearance from 20 to 24 cm. These remain planning
heuristics, not hard collision guarantees.

The thrust reference previously used the desired force directly as a command,
although the prediction model applies a calibrated gain and offset. Racing
feedforward and hover fallback now use the inverse mapping. A regression test
checks hover and tilted acceleration against the actual symbolic dynamics.
Clearance parameters are included in the plan-cache key. Failed racing episodes
also report the last valid observed pose and tracking error as CI annotations,
without changing flight behavior.

| Screening | Seeds | Finishes | Mean successful lap |
| --- | --- | ---: | ---: |
| Previous 8-second MPC | 0?9 | 9/10 | 8.28 s |
| Clearance + thrust revision | 0?9 | 9/10 | 8.66 s |
| Revision, held-out seeds | 100?119 | 18/20 | 8.59 s |

The reproduced seed-4 collision is fixed, but a different screening seed failed;
this is not a claim of universal improvement on every initial state. The complete
[per-race results](mpc_reliability_results.csv) include failures. All 12 relevant
unit tests and both Ruff checks pass. Central competition validation of controller commit `0bb30db` subsequently
finished **17/20 races (85%)**, with a mean successful lap of **8.47765 s**.
The leaderboard was updated. This confirms the fast MPC now passes the actual
competition server; failures remain possible.

[Central evaluation](https://github.com/learnsyslab/lsy_drone_racing/actions/runs/39978593607).

## Submission verification

The submitted commit `bd317a8` passed the protected-file check and username
registration, but its central flight evaluation failed: 15 of 20 races failed.
The leaderboard update was skipped. See the
[central evaluation](https://github.com/learnsyslab/lsy_drone_racing/actions/runs/39849044523).
The earlier local result of 14/20 successes did not generalize to that evaluation.

## Retained correction

`RaceTrajectory._optimize` previously cached a trajectory by geometry alone.
Its optimization also depends on the drone's initial position and velocity.
Consequently a plan optimized for the first race could be reused for a different
initial state in later races. The cache now includes both initial-state vectors.
A regression test checks reuse for identical inputs and recomputation after
changing either vector. This is a correctness fix, not a claim that the cache
alone explains every central failure.

## Independent-plan evaluation

After the cache correction, the unchanged 20-race evaluator with
`TIME_SCALE = 1.8` finished 15/20 races (75%), with a mean successful lap of
8.38 seconds. A separate batch at `TIME_SCALE = 1.4` completed 12/20 races
(60%) with a mean successful lap of 6.50 seconds. These historical batches used
1.8 as the default for greater reliability margin. They are fresh randomized batches, not paired comparisons with
the earlier 70% batch. It is a local pass, not a verified central pass.

## Selected experiments

These are screening runs in the unchanged randomized level-2 environment with
the attitude interface. They are not central competition certifications. Small
seed sets are useful for rejecting poor candidates, not establishing reliability.

| Candidate | Screening result | Decision |
| --- | --- | --- |
| Position-based reference progress, time scale 1.3 | 5/10 finishes; mean 6.59 s over finishes | Marginal reliability; not selected |
| Nonlinear gate-frame and pole clearance in the MPC | 0/3 finishes | Reject this implementation; high solver cost |
| Analytic-gradient planner with attitude feasibility penalties | Candidate paths around 4 s; 0/3 finishes at aggressive timing | Planning speed does not establish flight feasibility |
| Nonlinear contouring MPC, revised takeoff and costs | 1/3 finishes; successful lap 6.98 s | Experimental; not selected |
| Higher tracking gains, learned discrete attitude model, extra rate damping | No reliable fast candidate in the three-seed screenings | Not selected |

The three-seed screenings used 1337, 42 and 7. The ten-seed progress test added
100 through 106. Experiments used public observations and issued actions; no
hidden simulator state, physics, track, or competition script was changed.
Experimental controllers remain in ignored local `.pixi` files and are not
selected by `config/level2.toml`.

## Algorithm direction

[Model predictive contouring control](https://arxiv.org/abs/2108.13205) optimizes
progress along a spatial path rather than imposing fixed arrival times.
[MPCC++](https://arxiv.org/abs/2403.17551) combines that approach with spatial track
constraints and learned model residuals. These are relevant research directions,
but the local prototype is not a reproduction of those papers and did not
establish their guarantees or performance.

Further work should validate a model against held-out aggressive flights, enforce
clearance through the whole gate crossing, account for uncertainty before a gate
enters sensor range, and test multiple independent 20-race batches. Online
trajectory repair still sometimes exceeds the 20 ms control period. Three-second
performance and a successful central evaluation must be measured before claiming
either outcome.

## Aggressive MPC follow-up — 3 November 2026

The selected default now uses `TIME_SCALE = 1.45` and analytic nominal
initialization, retaining the existing section retiming, finish feasibility
check, planner clearances, MPC weights, horizon, hard actuator limits, physics,
scoring and gate validation. It achieved **8/10** finishes on fresh seeds
320–329 (80%); successful flights averaged **6.73 s**, with a fastest finish of
**6.62 s**. The three-seed screen (310–312) was 2/3. These are local randomized
simulations, not a population guarantee or a three-second result.

Three profiles were screened on the same seeds: the selected previous aggressive
profile (1.45 scale, analytic initialization), section-specific Gate 3 timing
(2/3, 7.21 s mean successful time), and bounded soft gate-plane constraints
(2/3, 7.15 s). The bounded profile worsened tracking and solver latency and was
not selected. Its duplicate hard outer rows correctly bounded the four gate
plane slacks, but these constraints do **not** encode frame or pole clearance.
Frame/pole avoidance remains a sampled planner-objective penalty, not an MPC
constraint. Actual collision and gate-crossing rules were untouched.

Across the eight successful validation flights, the longest mean intervals were
takeoff→Gate 1 and Gate 2→Gate 3 (both about **1.47 s**). Both validation
failures stopped after two credited passages while approaching Gate 3; that
approach is now the primary reliability bottleneck. More aggressive timing is
not justified until lateral braking and full-body clearance there are improved.
Candidate profiles and per-seed telemetry are retained in
`.pixi/adr_level2_candidates_20261103/`.
