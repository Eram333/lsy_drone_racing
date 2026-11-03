# Level 2 MPC optimization experiment — 3 November 2026

## Decision

**No tested change improved race time while preserving the measured success rate.**
All production-controller changes from this experiment were reverted. The four
controller/planner files are byte-identical (SHA-256 checked) to the working-tree
copies saved before the experiment, including all pre-existing uncommitted work.
The 3–4 second target was not achieved. No commit or push was made.

## Paired validation

Ten additional Level 2 seeds, **200–209**, were selected before validation and
were not used for the three optimization iterations. Times below are means over
successful races, not wall-clock program runtime. Failures are included in success
and contact counts; they are not counted as fast finishes.

| Version | Successful race time | Success | Contact collisions | Median initialization | MPC solve p95 | Mean per-race tracking RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original working tree, scale 1.7 | 7.867 s | 9/10 | 1 | 15.736 s | 13.60 ms | 7.04 cm |
| Analytic nominal initialization only, scale 1.7 — rejected | 7.891 s | 7/10 | 3 | 0.487 s | 13.49 ms | 7.60 cm |
| Analytic initialization + scale 1.45 — rejected | 6.760 s | 6/10 | 4 | 0.432 s | 14.06 ms | 7.12 cm |

There were no boundary terminations, timeouts, or observed uncredited target-plane
crossings in these batches. A collision before crossing the plane is counted as
a contact failure, not as an additional gate miss. All credited gate counts and
passage times are retained in the raw results.

For the six seeds completed by both the original and the faster candidate, mean
time fell from 7.863 to 6.760 s (**14.0%**), but the extra three contact failures
make that candidate unsuitable as a reliability-preserving replacement. The
sample sizes do not establish population-level success probabilities.

## Three iterations

Screening seeds were 0, 1 and 101; all variants used the same explicit reset seeds.

| Version | Screening results | Interpretation |
| --- | --- | --- |
| Original | 9.30, 7.84, 7.88 s; 3/3 finishes | Mean 8.34 s |
| 1: use the existing analytic cost/gradient for nominal L-BFGS-B initialization | 7.92, 7.80, 7.88 s; 3/3 | Faster initialization and a better-looking screening result; later failed validation |
| 2: also reduce global time scale from 1.7 to 1.45 | 6.78, 6.66, 6.72 s; 3/3 | Faster screening; success fell to 6/10 on additional seeds |
| 3: scale 1.2, with first-leg time/speed compensation | 5.82 s on seed 0; contact failures on seeds 1 and 101 | Only 1/3 finishes; rejected without further tuning |

Iteration 3 used section factors `(1.15, 1, 1, 0.95)` and crossing-speed factors
`(0.83, 1, 1, 1.05)`. Similar takeoff duration does not imply an identical takeoff
trajectory: shared endpoint accelerations and the following horizon also change.

Analytic derivatives reduced median initialization about **96.9%** on the additional
seeds, but changed the optimizer's resulting trajectories. Matching the cost and
its derivative does not guarantee matching a local optimum or flight reliability.
This benefit was therefore not retained in the default controller.

## Bottlenecks and next work

- **Timing versus tracking:** the original seed-0 nominal schedule was 9.429 s and
  its flight 9.30 s. Faster schedules do shorten flights, but are not reliably
  trackable. First-leg tilt commands already reached the existing 1 rad limit at
  scale 1.45; simply increasing tracking weights does not create more authority.
- **Fourth-gate approach/braking:** three of the four scale-1.45 validation failures
  occurred approaching the fourth physical gate. Last valid lateral offsets were
  about 15–20 cm, with lateral tracking errors around 8–12 cm and no reported
  repeated MPC solve failures. The opening half-width is 20 cm. These measurements
  do not justify shrinking frame or pole margins.
- **Model and replanning:** the MPC uses the identified `so_rpy` model while the
  configured simulation remains first-principles. Nominal planning penalizes
  sampled force, speed and jerk, but does not enforce the complete rotational/
  motor feasibility envelope. Online repair changes a global reference with soft
  position/velocity anchoring and fixed segment deadlines; reference C2 joins do
  not ensure a C2-continuous switch between old and repaired trajectories.
- **Next experiment:** keep the reliable path shape initially, measure lateral
  braking and actuator saturation around gate 4, and test section-specific timing
  rather than another global compression. Validate actuator-feasible, continuous
  remaining-path repairs before shortening the tight turns. There is no measured
  reason here to switch to MPCC or rewrite the controller.

MPC weights, the 20-stage/0.8 s horizon, frame/pole clearance settings, physics,
race rules and scoring were not changed. The final controller is the original,
not the apparently faster three-seed winner.

## Runtime and verification

Control-step p95 across the ten seeds was 20.18 ms original, 19.69 ms analytic-only,
and 20.23 ms faster. Occasional deadline misses remain. The analytic-only run on
seed 207 contains a **392.683 s wall-clock interval inside one solve**; its cause
was not established. It remains in the raw data and contaminates mean/max runtime;
it is not evidence of ordinary solver computation cost. The p95 values above
retain that observation. No hard real-time improvement is claimed.

Initialization comparisons use cached solver/native-library setup; cold JAX/C
compilation is separate. An original seed-0 replay exactly reproduced the 9.30 s
flight and gate timings after terminal-reason instrumentation was added.

The existing 19 focused unit tests passed for the analytic implementation,
including independent NumPy/finite-difference gradient comparisons, signed gates,
cache keys, C2 joins, retiming and failed-repair rejection. Ruff and editor
diagnostics were clean. The project simulator/evaluator and environment source
files were not edited. After restoring the original controller, all 19 focused
tests passed again; Ruff, editor diagnostics and `git diff --check` were clean.

## Reproduction and retained data

[Benchmark runner](../../benchmarks/racing_mpc.py) wraps the existing
`scripts.sim.simulate` loop. Only reset seeds and headless rendering are set in
memory. Terminal contacts/bounds are observed without changing the existing
disable decision. Controllers still receive only normal public observations.
NumPy traces contain the documented `trace_columns` plus solve/control/repair
wall-clock arrays. Core-source hashes are saved with the results.

This machine's usable existing environment is the detached Pixi CPU environment;
the workspace-local Python package contents were incomplete. No packages were
installed to work around that.

```powershell
$adrPython = 'C:/Users/narge/AppData/Local/lsy-team-pixi-cache/envs/lsy_drone_racing-15404871495157294331/envs/default/python.exe'
& $adrPython benchmarks/racing_mpc.py --seeds 0,200,201,202,203,204,205,206,207,208,209 --output .pixi/adr_20261102/replay.json
```

The default now reproduces the original restored controller. Preserved experimental
sources can be selected with `--source-dir .pixi/adr_20261102/iteration2`; add
`--scale 1.7` for analytic-only initialization. The rejected third profile is in
`.pixi/adr_20261102/iteration3`. The saved original is in
`.pixi/adr_20261102/baseline`.

Raw validation: [original](../../.pixi/adr_20261102/baseline_validation.json),
[analytic-only](../../.pixi/adr_20261102/analytic_validation.json),
[faster rejected candidate](../../.pixi/adr_20261102/optimized_validation.json).
Screening: [original](../../.pixi/adr_20261102/baseline_screen.json),
[iteration 1](../../.pixi/adr_20261102/iteration1_screen.json),
[iteration 2](../../.pixi/adr_20261102/iteration2_screen.json),
[iteration 3](../../.pixi/adr_20261102/iteration3_screen.json).
