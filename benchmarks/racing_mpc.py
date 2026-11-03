"""Fixed-seed telemetry around the unmodified project simulation/evaluation loop.

Run with the existing Windows launcher environment, e.g. Python benchmarks/racing_mpc.py
--seeds 0,1,101 --output .pixi/adr_20261102/baseline.json.
No physics, scoring, track configuration or controller observations are changed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import logging
import os
import runpy
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SCIPY_ARRAY_API", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")


def main() -> None:
    """Instrument scripts.sim.simulate without modifying its flight loop."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0,1,101")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float)
    parser.add_argument("--weights", help="16 comma-separated state/input weights")
    parser.add_argument("--feedforward", type=float, default=0)
    parser.add_argument("--feedforward-progress", type=int)
    parser.add_argument(
        "--profile", type=Path, help="Explicit controller/trajectory class settings"
    )
    parser.add_argument(
        "--source-dir", type=Path, help="Saved controller sources for paired baselines"
    )
    args = parser.parse_args()
    # Reuse the existing Windows template/build compatibility shim.
    if os.name == "nt":
        prefix = Path(sys.prefix)
        sys.path.insert(0, str(ROOT / "acados/interfaces/acados_template"))
        os.environ["PATH"] = os.pathsep.join(
            [str(prefix / "Library/bin"), str(ROOT / "acados/bin"), os.environ["PATH"]]
        )
        os.environ.setdefault("ACADOS_SOURCE_DIR", str(ROOT / "acados"))
        os.environ.setdefault("ACADOS_INSTALL_DIR", str(ROOT / "acados"))
        os.environ.setdefault("TERA_PATH", str(ROOT / "acados/bin/t_renderer.exe"))
        runpy.run_path(str(ROOT / ".pixi/run_level2_windows.py"), run_name="adr_windows_setup")

    import jax
    import jax.numpy as jnp
    import numpy as np
    from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

    import lsy_drone_racing.envs.race_core as race_core
    import scripts.sim as simulation
    from lsy_drone_racing.utils import load_config, load_controller

    logging.basicConfig(level=logging.ERROR)
    seeds = [int(seed) for seed in args.seeds.split(",")]
    config = load_config(ROOT / "config/level2.toml")
    config.env.seed = seeds[0]
    config.sim.render = False
    source_dir = args.source_dir or ROOT / "lsy_drone_racing/control"
    if args.source_dir:
        for name in ("attitude_mpc", "racing_trajectory", "trajectory_gradient"):
            module_name = "lsy_drone_racing.control." + name
            spec = importlib.util.spec_from_file_location(module_name, source_dir / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            # Keep generated/native caches rooted in the actual project, not the backup folder.
            module.__file__ = str(ROOT / "lsy_drone_racing/control" / (name + ".py"))
            spec.loader.exec_module(module)
    controller_cls = load_controller(source_dir / config.controller.file)
    profile = json.loads(args.profile.read_text()) if args.profile else {}
    trajectory_module = sys.modules.get("lsy_drone_racing.control.racing_trajectory")
    if trajectory_module is None:
        trajectory_module = importlib.import_module(
            "lsy_drone_racing.control.racing_trajectory"
        )
    trajectory_cls = trajectory_module.RaceTrajectory
    for cls, settings in (
        (controller_cls, profile.get("controller", {})),
        (trajectory_cls, profile.get("trajectory", {})),
    ):
        for name, value in settings.items():
            if not name.isupper() or not hasattr(cls, name):
                raise ValueError(f"Unknown profile setting: {name}")
            setattr(cls, name, tuple(value) if isinstance(value, list) else value)
    attitude_module = sys.modules["lsy_drone_racing.control.attitude_mpc"]
    for cache in (ROOT / "c_generated_code").glob("mpc_*"):
        if (cache / "acados_ocp_solver_basic_example_mpc.dll").exists() and (
            cache / "ocp.json"
        ).exists():
            attitude_module._built_solvers.add(cache.name.removeprefix("mpc_"))
    if args.scale is not None:
        controller_cls.TIME_SCALE = args.scale
    results = []
    active = {}
    original_disabled = race_core._disabled_drones

    def record_terminal(contacts: np.ndarray, bounds: np.ndarray) -> None:
        active["contact_collision"] = bool(contacts)
        active["boundary_failure"] = bool(bounds)

    def observed_disabled(pos: object, contacts: object, data: object) -> object:
        """Observe the existing termination decision without changing its result."""
        disabled = original_disabled(pos, contacts, data)
        contact = jnp.any(contacts[:, :, None] & data.contact_masks)
        off_platform = jnp.any(jnp.abs(pos[..., :2] - data.takeoff_pos[..., :2]) > 0.02, axis=-1)
        bounds = jnp.any(jnp.any(pos < data.pos_limit_low, axis=-1) & off_platform)
        bounds |= jnp.any(pos > data.pos_limit_high)

        def report() -> None:
            jax.debug.callback(record_terminal, contact, bounds, ordered=True)

        jax.lax.cond(jnp.any(disabled & ~data.disabled_drones), report, lambda: None)
        return disabled

    class SeededNumpy(JaxToNumpy):
        def reset(self, **kwargs: object) -> tuple[dict, dict]:
            seed = seeds[len(results)]
            active.clear()
            active.update(
                seed=seed,
                controls=[],
                solves=[],
                repairs=[],
                errors=[],
                rows=[],
                crossings=[],
                misses=0,
                terminated=False,
                truncated=False,
                contact_collision=False,
                boundary_failure=False,
                finish_check_s=0.0,
                soft_max=np.zeros(4),
                slack_max=np.zeros(4),
                hard_violation_max=0.0,
                actuator_violation_max=0.0,
                corridor_steps=0,
                solver_statuses={},
            )
            return super().reset(seed=seed)

        def step(self, action: np.ndarray) -> tuple:
            output = super().step(action)
            obs, _, terminated, truncated, _ = output
            if terminated:
                jax.effects_barrier()
            active.update(terminated=bool(terminated), truncated=bool(truncated), final=obs)
            return output

    class SolverTelemetry:
        def __init__(self, solver: object):
            self.solver = solver
            self.constraints = {}

        def __getattr__(self, name: str) -> object:
            return getattr(self.solver, name)

        def solve(self) -> int:
            started = time.perf_counter()
            status = self.solver.solve()
            active["solves"].append(time.perf_counter() - started)
            statuses = active["solver_statuses"]
            statuses[str(status)] = statuses.get(str(status), 0) + 1
            return status

        def constraints_set(
            self, stage: int, name: str, value: np.ndarray, **kwargs: object
        ) -> None:
            self.constraints[stage, name] = np.array(value, copy=True)
            self.solver.constraints_set(stage, name, value, **kwargs)

        def set(self, stage: int, name: str, value: np.ndarray) -> None:
            if (
                name == "yref"
                and args.feedforward
                and len(value) >= 15
                and (
                    args.feedforward_progress is None
                    or active.get("gate_progress") == args.feedforward_progress
                )
            ):
                # Optional controlled screening of rotational-model inversion.
                value = value.copy()
                params = active["params"]
                command = (
                    -(params["rpy_coef"] * value[3:6] + params["rpy_rates_coef"] * value[9:12])
                    / params["cmd_rpy_coef"]
                )
                value[12:15] += args.feedforward * (command - value[12:15])
            self.solver.set(stage, name, value)

    class MeasuredController(controller_cls):
        def __init__(self, obs: dict, info: dict, cfg: dict):
            started = time.perf_counter()
            super().__init__(obs, info, cfg)
            active["initialization_s"] = time.perf_counter() - started
            active["params"] = self._params
            self._solver = SolverTelemetry(self._solver)
            if args.weights:
                weights = np.array([float(x) for x in args.weights.split(",")])
                if weights.shape != (16,) or np.any(weights <= 0):
                    raise ValueError("Expected 16 positive weights")
                for stage in range(self._N):
                    self._solver.cost_set(stage, "W", np.diag(weights))
                self._solver.cost_set(self._N, "W", np.diag(weights[:12]))
            active["planned_durations"] = self._trajectory._variables[
                : len(obs["gate_sequence"])
            ].tolist()
            self._old_progress = 0
            self._old_pos = np.array(obs["pos"])
            repair = self._trajectory._repair_plan

            def measured_repair(*values: object) -> None:
                started = time.perf_counter()
                repair(*values)
                active["repairs"].append(time.perf_counter() - started)

            self._trajectory._repair_plan = measured_repair
            if hasattr(self._trajectory, "_retime_finish"):
                finish = self._trajectory._retime_finish

                def measured_finish() -> None:
                    started = time.perf_counter()
                    finish()
                    active["finish_check_s"] += time.perf_counter() - started

                self._trajectory._retime_finish = measured_finish

        def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
            started = time.perf_counter()
            active["gate_progress"] = int(obs["n_gates_passed"])
            action = super().compute_control(obs, info)
            active["controls"].append(time.perf_counter() - started)
            error = float(np.linalg.norm(self._last_reference_position - obs["pos"]))
            active["errors"].append(error)
            passage = min(int(obs["n_gates_passed"]), len(obs["gate_sequence"]) - 1)
            target_gate = int(obs["gate_sequence"][passage])
            active["rows"].append(
                np.r_[
                    self._tick * self._dt,
                    obs["n_gates_passed"],
                    obs["pos"],
                    obs["vel"],
                    self._last_reference_position,
                    action,
                    self._reference_acceleration[0],
                    self._trajectory._time,
                    obs["quat"],
                    obs["ang_vel"],
                    target_gate,
                    obs["gates_pos"][target_gate],
                    obs["gates_quat"][target_gate],
                    obs["gate_sequence_direction"][passage],
                ]
            )
            self._old_pos = np.array(obs["pos"])
            self._old_progress = int(obs["n_gates_passed"])
            stages = getattr(self, "_gate_constraint_stages", None)
            if stages is None:
                stages = (
                    [] if self._gate_constraint_stage is None else [self._gate_constraint_stage]
                )
            for stage in stages:
                constraints = self._solver.constraints
                predicted = constraints[stage, "C"] @ self._solver.get(stage, "x")
                violation = np.maximum(
                    np.maximum(
                        constraints[stage, "lg"] - predicted, predicted - constraints[stage, "ug"]
                    ),
                    0,
                )
                active["soft_max"] = np.maximum(active["soft_max"], violation[:4])
                slack = np.maximum(self._solver.get(stage, "sl"), self._solver.get(stage, "su"))
                active["slack_max"] = np.maximum(active["slack_max"], slack[:4])
                if len(violation) > 4:
                    active["hard_violation_max"] = max(
                        active["hard_violation_max"], float(violation[4:].max())
                    )
                active["corridor_steps"] += 1
            active["actuator_violation_max"] = max(
                active["actuator_violation_max"],
                float(
                    np.maximum(
                        self._ocp.constraints.lbu - action, action - self._ocp.constraints.ubu
                    ).max()
                ),
                0,
            )
            return action

        def step_callback(
            self,
            action: np.ndarray,
            obs: dict,
            reward: float,
            terminated: bool,
            truncated: bool,
            info: dict,
        ) -> bool:
            progress = int(obs["n_gates_passed"])
            if progress > self._old_progress:
                active["crossings"].append(
                    {
                        "passage": progress,
                        "time_s": self._tick * self._dt,
                        "speed_m_s": float(np.linalg.norm(obs["vel"])),
                    }
                )
            # An observed forward crossing of the target plane without credited progress.
            if (
                not terminated
                and progress == self._old_progress
                and progress < len(obs["gate_sequence"])
            ):
                from scipy.spatial.transform import Rotation

                gate = int(obs["gate_sequence"][progress])
                normal = Rotation.from_quat(obs["gates_quat"][gate]).as_matrix()[:, 0]
                normal *= obs["gate_sequence_direction"][progress]
                centre = obs["gates_pos"][gate]
                if (self._old_pos - centre) @ normal < 0 <= (obs["pos"] - centre) @ normal:
                    active["misses"] += 1
            return super().step_callback(action, obs, reward, terminated, truncated, info)

        def episode_callback(self) -> None:
            super().episode_callback()
            obs = active["final"]
            success = int(obs["n_gates_passed"]) == len(obs["gate_sequence"])
            result = {
                "seed": active["seed"],
                "success": success,
                "flight_s": (self._tick - 1) * self._dt,
                "gates": int(obs["n_gates_passed"]),
                "collision_or_bounds": active["terminated"] and not success,
                "contact_collision": active["contact_collision"],
                "boundary_failure": active["boundary_failure"],
                "timeout": active["truncated"],
                "observed_gate_misses": active["misses"],
                "init_s": active["initialization_s"],
                "planned_durations": active["planned_durations"],
                "final_planned_durations": self._trajectory._variables[
                    : len(obs["gate_sequence"])
                ].tolist(),
                "finish_factor_used": getattr(self._trajectory, "_finish_factor_used", 1.0),
                "finish_check_ms": active["finish_check_s"] * 1000,
                "crossings": active["crossings"],
                "repair_count": len(active["repairs"]),
                "repair_total_s": sum(active["repairs"]),
                "tracking_rms_m": float(np.sqrt(np.mean(np.square(active["errors"])))),
                "tracking_p95_m": float(np.percentile(active["errors"], 95)),
                "over_20ms": int(np.sum(np.array(active["controls"]) > self._dt)),
                "soft_corridor_max": active["soft_max"].tolist(),
                "solver_slack_max": active["slack_max"].tolist(),
                "hard_corridor_violation_max": active["hard_violation_max"],
                "actuator_violation_max": active["actuator_violation_max"],
                "corridor_steps": active["corridor_steps"],
                "solver_statuses": active["solver_statuses"],
            }
            for label in ("controls", "solves"):
                values = np.array(active[label]) * 1000
                result[label + "_ms"] = {
                    "mean": float(values.mean()),
                    "p95": float(np.percentile(values, 95)),
                    "max": float(values.max()),
                }
            results.append(result)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.output.with_name(args.output.stem + f"_seed{active['seed']}.npz"),
                rows=np.array(active["rows"]),
                control_s=np.array(active["controls"]),
                solve_s=np.array(active["solves"]),
                repair_s=np.array(active["repairs"]),
                trajectory_variables=self._trajectory._variables,
                trajectory_coefficients=self._trajectory._coefficients,
                trajectory_points=self._trajectory._points,
                trajectory_start_velocity=self._trajectory._start_velocity,
                trajectory_gates=self._trajectory._gates,
                trajectory_rotations=self._trajectory._rotations,
                trajectory_obstacles=self._trajectory._obstacles,
                trajectory_sequence=self._trajectory._sequence,
                trajectory_directions=self._trajectory._directions,
            )
            payload = {
                "seeds": seeds,
                "runtime": {
                    "python": sys.version,
                    "executable": sys.executable,
                    "numpy": np.__version__,
                    "jax": jax.__version__,
                    "backend": jax.default_backend(),
                },
                "trace_columns": [
                    "time_s",
                    "gates_passed",
                    "x",
                    "y",
                    "z",
                    "vx",
                    "vy",
                    "vz",
                    "reference_x",
                    "reference_y",
                    "reference_z",
                    "command_roll",
                    "command_pitch",
                    "command_yaw",
                    "command_thrust",
                    "reference_ax",
                    "reference_ay",
                    "reference_az",
                    "reference_clock_s",
                    "qx",
                    "qy",
                    "qz",
                    "qw",
                    "omega_x",
                    "omega_y",
                    "omega_z",
                    "target_gate",
                    "target_x",
                    "target_y",
                    "target_z",
                    "target_qx",
                    "target_qy",
                    "target_qz",
                    "target_qw",
                    "target_direction",
                ],
                "config_sha256": hashlib.sha256(
                    (ROOT / "config/level2.toml").read_bytes()
                ).hexdigest(),
                "options": vars(args)
                | {
                    "output": str(args.output),
                    "source_dir": str(source_dir),
                    "profile": str(args.profile) if args.profile else None,
                },
                "profile": profile,
                "source_sha256": {
                    name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
                    for name in (
                        "level2_mpc.py",
                        "racing_trajectory.py",
                        "trajectory_gradient.py",
                        "attitude_mpc.py",
                    )
                },
                "results": results,
            }
            args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(json.dumps(result), flush=True)

    with (
        patch.object(simulation, "load_config", return_value=config),
        patch.object(simulation, "load_controller", return_value=MeasuredController),
        patch.object(simulation, "JaxToNumpy", SeededNumpy),
        patch.object(race_core, "_disabled_drones", observed_disabled),
    ):
        simulation.simulate(config="level2.toml", n_runs=len(seeds), render=False)


if __name__ == "__main__":
    main()
