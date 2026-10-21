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

The controller uses the supplied acados nonlinear drone model with a one-second,
25-stage prediction horizon. It replans from sensor observations, respects signed
gate order, routes around obstacles and gate frames, and clears each frame before
turning toward the next gate. Clearance is enforced by the route planner rather
than hard collision constraints inside the MPC, so randomized races can still fail.

On Windows, Pixi activation builds acados and repairs GCC specs that contain an
unquoted installation path with spaces. The first run compiles the generated solver;
subsequent episodes in the same process reuse its DLL.

Local Windows CPU validation: the 20-race evaluation completed 19 races successfully
(95%), averaging 25.74 seconds over successful races. Seeds 1337, 42 and 7 completed
in 27.12, 26.16 and 24.80 seconds respectively. Five planner regression tests pass.
These are baseline measurements, not a guarantee for future random seeds. Some
geometry replans exceed the 20 ms control period.
