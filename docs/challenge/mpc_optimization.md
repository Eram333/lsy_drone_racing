# MPC optimization notes — 23 October 2026

The three-second target has not been achieved. A planned trajectory duration is
not an achieved lap time, and passing lint is not passing the competition flight
evaluation.

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
8.38 seconds. This is a fresh randomized batch, not a paired comparison with
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

