# Simulation

The first part of the challenge is to complete the race track in simulation. To that end, we provide a high-fidelity simulator with interfaces that are identical to the real-world setup. We use [crazyflow](https://github.com/learnsyslab/crazyflow/tree/main) as the underlying simulator to model the physics of the drone and to render the environment.

## Running the Simulation

Once you have installed the `lsy_drone_racing` package, you should be able to start a simulation inside a `pixi shell` with

```bash
python scripts/sim.py
```

The simulation script has a built-in CLI that allows you to set a few useful arguments. For example, if you want the simulation to render its output and run more than just a single race, you can pass these arguments with

```bash
python scripts/sim.py -r -n 10
```

This will render the simulation (`-r` or `--render`) and run 10 races in succession (`-n` or `--n_runs`). For a list of all arguments, use `-h` or `--help`.

By default, rendering follows the `render` setting in the `[sim]` section of the config file. Use `--render=True` or `--render=False` to override it for the current run without modifying the configuration file.

## Switching Between Configurations

You can choose which configuration to use by changing the `--config` command line option. For example, to run the example controller on the hardest scenario, use the following command:

```bash
python scripts/sim.py --config level3.toml
```

To use your own controller, you can pass the path to your controller script as the `--controller` argument. For example:

```bash
python scripts/sim.py --config level3.toml --controller my_controller.py
```

!!! note
    You can also write your controller file directly into the config files located in the `config` folder. That way, you don't need to specify the controller script when running the simulation, and the controller will be used in the automated challenge evaluation.

## Level 2 MPC

`config/level2.toml` selects `level2_mpc.py` with the attitude interface. Run it with:

```powershell
pixi run python scripts/sim.py --config level2.toml
```

Use `--render=False` for a run without graphics, or `pixi run evaluate` for the
competition's 20-episode evaluation. The controller implements the `Controller`
interface in a separate file; `control/controller.py` remains the common base class.

The racing profile uses the supplied acados nonlinear drone model with a 0.8-second,
20-stage prediction horizon. A continuous quintic reference carries velocity through
gate crossings, with attitude and acceleration feedforward. The controller replans
from public sensor observations and adds soft constraints for the current signed
gate opening. Sampled obstacle and frame clearance remains a planning heuristic,
so randomized races can still fail. The simulator, track, and competition scripts
are unchanged.

`Level2MPC.TIME_SCALE` controls the trajectory duration: a smaller value requests
faster motion and can sharply reduce reliability. `Level2MPC.RACING = False`
restores the conservative visibility-graph planner and original MPC settings.

On Windows, Pixi activation builds acados and repairs GCC specs that contain an
unquoted installation path with spaces. The first run compiles the generated solver;
subsequent episodes in the same process reuse its DLL.

Local Windows CPU validation with the unchanged 20-race evaluator:

| Profile | Mean successful lap | Finished races |
| --- | ---: | ---: |
| Original conservative planner | 25.74 s | 19/20 (95%) |
| Racing, `TIME_SCALE = 1.8` (default) | 8.30 s | 14/20 (70%) |
| Racing, `TIME_SCALE = 1.6` | approximately 7.4 s | 9/20 (45%, fails threshold) |
| Racing, `TIME_SCALE = 1.4` | approximately 6.5 s | 6/20 (30%, fails threshold) |

The default reduces the measured mean successful lap time by approximately 68%,
but gives up reliability. Three-second laps have **not** been achieved. Faster
settings are experimental and did not pass the required 50% success threshold.
These local results are not a guarantee for future randomized races.

Ten planner and gate-corridor regression tests pass. Sensor-triggered trajectory
optimization sometimes exceeds the 20 ms control period by over a second;
initial trajectory optimization also takes several seconds. The lap times above
are simulated flight times, not wall-clock execution times. This profile needs
further work before use with a physical drone or strict real-time execution.
