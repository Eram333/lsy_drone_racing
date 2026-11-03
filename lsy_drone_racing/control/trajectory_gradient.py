"""Cached analytic derivatives of the existing racing trajectory cost."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import casadi as ca
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.racing_trajectory import RaceTrajectory

logger = logging.getLogger(__name__)


def _native_function(function: ca.Function) -> ca.Function:
    """Cache generated C under .pixi; use the interpreter if no C compiler is available."""
    compiler = shutil.which("gcc") or shutil.which("clang")
    if compiler is None:
        return function
    cache = Path(__file__).resolve().parents[2] / ".pixi" / "trajectory_costs"
    signature = ca.__version__ + sys.platform + "-O2" + function.serialize()
    digest = hashlib.sha256(signature.encode()).hexdigest()[:20]
    suffix = ".dll" if os.name == "nt" else ".so"
    library = cache / f"cost_{digest}{suffix}"
    try:
        cache.mkdir(parents=True, exist_ok=True)
        if not library.exists():
            logger.info("Compiling trajectory derivatives once before flight: %s", library.name)
            stem = f"cost_{digest}_{os.getpid()}"
            generator = ca.CodeGenerator(stem + ".c")
            generator.add(function)
            source = generator.generate(str(cache) + os.sep)
            temporary = cache / (stem + suffix)
            command = [compiler, "-O2", "-shared", source, "-o", str(temporary)]
            if os.name != "nt":
                command += ["-fPIC", "-lm"]
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
            if not library.exists():
                temporary.replace(library)
            logger.info("Compiled and cached trajectory derivatives: %s", library.name)
        return ca.external(function.name(), str(library))
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        logger.warning("Native trajectory compilation unavailable; using CasADi graph: %s", error)
        return function


def _norm(columns: ca.MX) -> ca.MX:
    """Return column norms with finite derivatives at zero."""
    return ca.sqrt(ca.sum1(columns**2) + 1e-24)


@lru_cache(maxsize=16)
def _cost_function(
    sequence: tuple[int, ...], directions: tuple[float, ...], n_gates: int, n_obstacles: int
) -> ca.Function:
    """Build once per track topology, with geometry and cost weights as inputs."""
    count = len(sequence)
    # Matrix graphs generate compact loops instead of enormous unrolled C functions.
    variables = ca.MX.sym("variables", 7 * count)
    start, start_velocity = ca.MX.sym("start", 3), ca.MX.sym("start_velocity", 3)
    start_acceleration = ca.MX.sym("start_acceleration", 3)
    gates = ca.MX.sym("gates", 3, n_gates)
    rotations = ca.MX.sym("rotations", 9, n_gates)
    obstacles = ca.MX.sym("obstacles", 3, n_obstacles)
    settings = ca.MX.sym("settings", 5)
    anchor = ca.MX.sym("anchor", count + 18)
    original = ca.MX.sym("original", 7 * count)
    coefficients = []
    sampled = [[] for _ in range(4)]
    samples = np.linspace(0, 1, 35)
    powers = np.arange(6)
    bases = []
    for derivative in range(4):
        factors = np.ones(6)
        for order in range(derivative):
            factors *= np.maximum(powers - order, 0)
        bases.append(ca.DM((samples[:, None] ** np.maximum(powers - derivative, 0) * factors).T))
    point0, velocity0, acceleration0 = start, start_velocity, start_acceleration
    for segment, gate in enumerate(sequence):
        duration = variables[segment]
        rotation = ca.reshape(rotations[:, gate], 3, 3)
        point1 = gates[:, gate]
        velocity1 = (
            variables[count + segment] * rotation[:, 0] * directions[segment]
            + variables[5 * count + 2 * segment] * rotation[:, 1]
            + variables[5 * count + 2 * segment + 1] * rotation[:, 2]
        )
        acceleration1 = variables[2 * count + 3 * segment : 2 * count + 3 * segment + 3]
        delta = point1 - point0
        v0, v1 = velocity0 * duration, velocity1 * duration
        a0, a1 = acceleration0 * duration**2, acceleration1 * duration**2
        polynomial = ca.horzcat(
            point0,
            v0,
            a0 / 2,
            10 * delta - 6 * v0 - 4 * v1 - 1.5 * a0 + 0.5 * a1,
            -15 * delta + 8 * v0 + 7 * v1 + 1.5 * a0 - a1,
            6 * delta - 3 * (v0 + v1) - 0.5 * a0 + 0.5 * a1,
        )
        coefficients.append(polynomial)
        for derivative, basis in enumerate(bases):
            sampled[derivative].append(polynomial @ basis / duration**derivative)
        point0, velocity0, acceleration0 = point1, velocity1, acceleration1
    positions, velocity, acceleration, jerk = [ca.horzcat(*parts) for parts in sampled]
    n_samples = count * len(samples)
    clearance = ca.MX.ones(1, n_samples) * 1e6
    centres = [[0, -0.28, 0], [0, 0.28, 0], [0, 0, -0.28], [0, 0, 0.28]]
    half_sizes = [[0.018, 0.08, 0.36]] * 2 + [[0.018, 0.36, 0.08]] * 2
    for gate in range(n_gates):
        rotation = ca.reshape(rotations[:, gate], 3, 3)
        local = rotation.T @ (positions - ca.repmat(gates[:, gate], 1, n_samples))
        for centre, half_size in zip(centres, half_sizes):
            delta = ca.fabs(local - ca.repmat(ca.DM(centre), 1, n_samples))
            delta -= ca.repmat(ca.DM(half_size), 1, n_samples)
            interior = ca.fmax(ca.fmax(delta[0, :], delta[1, :]), delta[2, :])
            signed = _norm(ca.fmax(delta, 0)) + ca.fmin(interior, 0)
            clearance = ca.fmin(clearance, signed)
    pole_cost = 0
    for obstacle in range(n_obstacles):
        distance = _norm(positions[:2, :] - ca.repmat(obstacles[:2, obstacle], 1, n_samples))
        violation = ca.if_else(
            positions[2, :] > obstacles[2, obstacle] + 0.08, 0, ca.fmax(settings[3] - distance, 0)
        )
        pole_cost += ca.sumsqr(violation)
    force = _norm(acceleration + ca.repmat(ca.DM([0, 0, 9.81]), 1, n_samples))
    cost = (
        ca.sum1(variables[:count])
        + 15000 * (ca.sumsqr(ca.fmax(settings[2] - clearance, 0)) + pole_cost)
        + 15000 * ca.sumsqr(ca.fmax(-positions[2, :], 0))
        + 20 * ca.sumsqr(ca.fmax(force - settings[0], 0)) / n_samples
        + 20 * ca.sumsqr(ca.fmax(2 - acceleration[2, :] - 9.81, 0)) / n_samples
        + 2 * ca.sumsqr(ca.fmax(_norm(velocity) - 4, 0)) / n_samples
        + settings[1] * ca.sumsqr(jerk) / n_samples
    )
    active = sum((c * anchor[i] for i, c in enumerate(coefficients)), ca.MX.zeros(3, 6))
    position = active @ anchor[count : count + 6]
    anchor_velocity = active @ anchor[count + 6 : count + 12]
    cost += settings[4] * (
        200 * ca.sumsqr(position - anchor[count + 12 : count + 15])
        + 2 * ca.sumsqr(anchor_velocity - anchor[count + 15 : count + 18])
        + 0.05 * ca.sumsqr(variables[2 * count :] - original[2 * count :])
    )
    function = ca.Function(
        "racing_cost_gradient",
        [
            variables,
            start,
            start_velocity,
            start_acceleration,
            gates,
            rotations,
            obstacles,
            settings,
            anchor,
            original,
        ],
        [cost, ca.gradient(cost, variables)],
        {"cse": True},
    )
    return _native_function(function)


class TrajectoryCost:
    """Prepare a parameterized cost graph before entering the control loop."""

    def __init__(self, planner: RaceTrajectory):
        """Cache derivatives by topology while keeping observed geometry dynamic."""
        self.function = _cost_function(
            tuple(map(int, planner._sequence)),
            tuple(map(float, planner._directions)),
            len(planner._gates),
            len(planner._obstacles),
        )

    def bind(
        self,
        planner: RaceTrajectory,
        anchor: NDArray | None = None,
        original: NDArray | None = None,
    ) -> Callable[[NDArray], tuple[float, NDArray]]:
        """Bind one sensor update, leaving only optimization variables as an argument."""
        count = len(planner._sequence)
        parameters = (
            planner._start,
            planner._start_velocity,
            getattr(planner, "_start_acceleration", np.zeros(3)),
            planner._gates.T,
            np.stack([rotation.ravel(order="F") for rotation in planner._rotations], axis=1),
            planner._obstacles.T,
            [
                planner.FORCE_LIMIT,
                planner.JERK_WEIGHT,
                planner.FRAME_CLEARANCE,
                planner.POLE_CLEARANCE,
                float(anchor is not None),
            ],
            np.zeros(count + 18) if anchor is None else anchor,
            planner._variables if original is None else original,
        )

        def evaluate(variables: NDArray) -> tuple[float, NDArray]:
            value, gradient = self.function(variables, *parameters)
            return float(value), np.asarray(gradient).ravel()

        return evaluate
