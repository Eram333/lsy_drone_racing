"""Continuous racing trajectories for the existing acados attitude MPC."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from numpy.typing import NDArray
logger = logging.getLogger(__name__)
_plan_cache: dict[bytes, NDArray] = {}


def quintic_coefficients(
    points: NDArray, velocity: NDArray, acceleration: NDArray, durations: NDArray
) -> NDArray:
    """Construct normalized-time quintics with continuous position, velocity and acceleration."""
    duration = durations[:, None]
    displacement = np.diff(points, axis=0)
    v0, v1 = velocity[:-1] * duration, velocity[1:] * duration
    a0, a1 = acceleration[:-1] * duration**2, acceleration[1:] * duration**2
    return np.stack(
        [
            points[:-1],
            v0,
            a0 / 2,
            10 * displacement - 6 * v0 - 4 * v1 - 1.5 * a0 + 0.5 * a1,
            -15 * displacement + 8 * v0 + 7 * v1 + 1.5 * a0 - a1,
            6 * displacement - 3 * (v0 + v1) - 0.5 * a0 + 0.5 * a1,
        ],
        axis=1,
    )


def sample_segments(
    coefficients: NDArray, durations: NDArray, samples: NDArray, derivative: int = 0
) -> NDArray:
    """Evaluate every polynomial segment on a shared normalized-time grid."""
    powers = np.arange(6)
    factors = np.ones(6)
    for order in range(derivative):
        factors *= np.maximum(powers - order, 0)
    basis = samples[:, None] ** np.maximum(powers - derivative, 0) * factors
    return np.einsum("skd,qk->sqd", coefficients, basis) / durations[:, None, None] ** derivative


def frame_clearance(positions: NDArray, gates: NDArray, rotations: NDArray) -> NDArray:
    """Distance to the four solid bars of each gate in its own coordinate frame."""
    local = np.einsum("nki,kij->nkj", positions[:, None] - gates[None], rotations)
    centres = np.array([[0, -0.28, 0], [0, 0.28, 0], [0, 0, -0.28], [0, 0, 0.28]])
    half_sizes = np.array([[0.018, 0.08, 0.36]] * 2 + [[0.018, 0.36, 0.08]] * 2)
    delta = np.abs(local[:, :, None] - centres) - half_sizes
    signed = np.linalg.norm(np.maximum(delta, 0), axis=-1) + np.minimum(np.max(delta, axis=-1), 0)
    return signed.min(axis=(1, 2))


class RaceTrajectory:
    """Plan from sensor geometry while tracking the configured signed gate order."""

    FORCE_LIMIT = 15.0
    JERK_WEIGHT = 0.00003
    FRAME_CLEARANCE = 0.15
    POLE_CLEARANCE = 0.24

    def __init__(self, obs: dict, time_scale: float = 1.5):
        """Optimize a nominal reference without changing any environment settings."""
        self._time = 0.0
        self._start = np.asarray(obs["pos"], dtype=float).copy()
        self._start_velocity = np.asarray(obs["vel"], dtype=float).copy()
        self._sequence = np.asarray(obs["gate_sequence"], dtype=int)
        self._directions = np.asarray(obs["gate_sequence_direction"], dtype=float)
        count = len(self._sequence)
        self._variables = np.r_[np.full(count, 1.1), np.full(count, 1.6), np.zeros(5 * count)]
        self._geometry(obs)
        self._geometry_key = self._key()
        self._optimize()
        self._variables[:count] *= time_scale
        self._variables[count : 2 * count] /= time_scale
        self._variables[2 * count : 5 * count] /= time_scale**2
        self._variables[5 * count :] /= time_scale
        self._coefficients = self._coefficients_for(self._variables)
        self._last_progress = 0
        logger.info("Continuous MPC reference: %.2f seconds", self._variables[:count].sum())

    def _key(self) -> bytes:
        """Identify sensor geometry so only new information triggers replanning."""
        return self._gates.tobytes() + self._rotations.tobytes() + self._obstacles.tobytes()

    def sample(self, obs: dict, offsets: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """Update the route and sample position, velocity and acceleration references."""
        old_position = self._reference(np.array([self._time]))[0]
        old_velocity = self._reference(np.array([self._time]), 1)[0]
        self._geometry(obs)
        if self._key() != self._geometry_key:
            self._geometry_key = self._key()
            self._repair_plan(old_position, old_velocity)
        self._coefficients = self._coefficients_for(self._variables)
        progress = int(obs["n_gates_passed"])
        boundaries = np.r_[0, np.cumsum(self._variables[: len(self._sequence)])]
        if progress < len(self._sequence):
            self._time = min(self._time, boundaries[progress + 1] + 0.1)
        if progress > self._last_progress:
            self._time = max(self._time, boundaries[progress])
            self._last_progress = progress
        times = self._time + offsets
        positions, velocities = self._reference(times), self._reference(times, 1)
        accelerations = self._reference(times, 2)
        return positions, velocities, accelerations

    def _geometry(self, obs: dict) -> None:
        """Refresh geometry only from the sensor-facing observation dictionary."""
        self._gates = np.asarray(obs["gates_pos"], dtype=float).copy()
        self._rotations = Rotation.from_quat(obs["gates_quat"]).as_matrix()
        normals = self._rotations[self._sequence, :, 0] * self._directions[:, None]
        self._normals = normals
        self._points = np.vstack([self._start, self._gates[self._sequence]])
        self._obstacles = np.asarray(obs["obstacles_pos"], dtype=float).copy()

    def _coefficients_for(self, variables: NDArray) -> NDArray:
        """Build a trajectory from segment times, crossing speeds and gate accelerations."""
        count = len(self._sequence)
        velocities = np.vstack([self._start_velocity, self._gate_velocities(variables)])
        accelerations = np.vstack([np.zeros(3), variables[2 * count : 5 * count].reshape(count, 3)])
        return quintic_coefficients(self._points, velocities, accelerations, variables[:count])

    def _gate_velocities(self, variables: NDArray) -> NDArray:
        """Optimize crossing angle while retaining a positive signed normal velocity."""
        count = len(self._sequence)
        tangent = variables[5 * count :].reshape(count, 2)
        return (
            variables[count : 2 * count, None] * self._normals
            + tangent[:, :1] * self._rotations[self._sequence, :, 1]
            + tangent[:, 1:] * self._rotations[self._sequence, :, 2]
        )

    def _objective(self, variables: NDArray) -> float:
        """Trade duration against sampled geometry and thrust-envelope violations."""
        count = len(self._sequence)
        times = variables[:count]
        coefficients = self._coefficients_for(variables)
        samples = np.linspace(0, 1, 35)
        positions = sample_segments(coefficients, times, samples).reshape(-1, 3)
        velocity = sample_segments(coefficients, times, samples, 1).reshape(-1, 3)
        acceleration = sample_segments(coefficients, times, samples, 2).reshape(-1, 3)
        jerk = sample_segments(coefficients, times, samples, 3).reshape(-1, 3)
        clearance = frame_clearance(positions, self._gates, self._rotations)
        pole_distance = np.linalg.norm(
            positions[:, None, :2] - self._obstacles[None, :, :2], axis=-1
        )
        pole_penalty = np.maximum(self.POLE_CLEARANCE - pole_distance, 0)
        pole_penalty[positions[:, None, 2] > self._obstacles[None, :, 2] + 0.08] = 0
        force = np.linalg.norm(acceleration + [0, 0, 9.81], axis=1)
        floor_penalty = np.maximum(-positions[:, 2], 0)
        return float(
            times.sum()
            + 15000
            * (
                np.square(np.maximum(self.FRAME_CLEARANCE - clearance, 0)).sum()
                + np.square(pole_penalty).sum()
            )
            + 15000 * np.square(floor_penalty).sum()
            + 20 * np.mean(np.maximum(force - self.FORCE_LIMIT, 0) ** 2)
            + 20 * np.mean(np.maximum(2 - acceleration[:, 2] - 9.81, 0) ** 2)
            + 2 * np.mean(np.maximum(np.linalg.norm(velocity, axis=1) - 4, 0) ** 2)
            + self.JERK_WEIGHT * np.mean(np.sum(jerk**2, axis=1))
        )

    def _optimize(self) -> None:
        """Optimize a feasible nominal trajectory before the control loop starts."""
        count = len(self._sequence)
        key = (
            self._start.tobytes()
            + self._start_velocity.tobytes()
            + self._gates.tobytes()
            + self._rotations.tobytes()
            + self._obstacles.tobytes()
            + self._sequence.tobytes()
            + self._directions.tobytes()
        )
        key += repr(
            (self.FORCE_LIMIT, self.JERK_WEIGHT, self.FRAME_CLEARANCE, self.POLE_CLEARANCE)
        ).encode()
        if key in _plan_cache:
            self._variables = _plan_cache[key].copy()
            return
        bounds = (
            [(0.35, 2.5)] * count
            + [(0.4, 3.5)] * count
            + [(-12, 12)] * (3 * count)
            + [(-2.5, 2.5), (0, 0)] * count
        )
        result = minimize(
            self._objective,
            self._variables,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 160, "ftol": 1e-8, "maxls": 30},
        )
        self._variables = result.x
        _plan_cache[key] = result.x.copy()
        logger.info("Trajectory optimization cost %.3f; converged=%s", result.fun, result.success)

    def _reference(self, times: NDArray, derivative: int = 0) -> NDArray:
        """Evaluate the sensor-updated trajectory at absolute race times."""
        durations = self._variables[: len(self._sequence)]
        boundaries = np.r_[0, np.cumsum(durations)]
        indices = np.minimum(
            np.searchsorted(boundaries[1:], times, side="right"), len(durations) - 1
        )
        normalized = np.clip((times - boundaries[indices]) / durations[indices], 0, 1)
        powers = np.arange(6)
        factors = np.ones(6)
        for order in range(derivative):
            factors *= np.maximum(powers - order, 0)
        basis = normalized[:, None] ** np.maximum(powers - derivative, 0) * factors
        values = np.einsum("nkd,nk->nd", self._coefficients[indices], basis)
        values = values / durations[indices, None] ** derivative
        beyond = times > boundaries[-1]
        elapsed = np.maximum(times[beyond] - boundaries[-1], 0)
        exit_velocity = self._gate_velocities(self._variables)[-1]
        if derivative == 0:
            values[beyond] = self._points[-1] + np.minimum(elapsed, 0.3)[:, None] * exit_velocity
        elif derivative == 1:
            values[beyond] = np.where((elapsed < 0.3)[:, None], exit_velocity, 0)
        else:
            values[beyond] = 0
        return values

    def _repair_plan(self, old_position: NDArray, old_velocity: NDArray) -> None:
        """Reshape the trajectory when newly observed geometry invalidates its nominal route."""
        count = len(self._sequence)
        original = self._variables.copy()
        durations = original[:count]
        boundaries = np.r_[0, np.cumsum(durations)]
        segment = min(np.searchsorted(boundaries[1:], self._time, side="right"), count - 1)
        tau = np.clip((self._time - boundaries[segment]) / durations[segment], 0, 1)
        basis = tau ** np.arange(6)
        dbasis = np.r_[0, np.arange(1, 6) * tau ** np.arange(5)] / durations[segment]

        def objective(variables: NDArray) -> float:
            coefficient = self._coefficients_for(variables)[segment]
            position = basis @ coefficient
            velocity = dbasis @ coefficient
            return (
                self._objective(variables)
                + 200 * np.sum((position - old_position) ** 2)
                + 2 * np.sum((velocity - old_velocity) ** 2)
                + 0.05 * np.sum((variables[2 * count :] - original[2 * count :]) ** 2)
            )

        result = minimize(
            objective,
            original,
            method="L-BFGS-B",
            bounds=[(v, v) for v in original[: 2 * count]]
            + [(max(-12, v - 6), min(12, v + 6)) for v in original[2 * count : 5 * count]]
            + [
                bound
                for lateral in original[5 * count :].reshape(count, 2)
                for bound in [(max(-2.5, lateral[0] - 0.6), min(2.5, lateral[0] + 0.6)), (0, 0)]
            ],
            options={"maxiter": 30, "maxls": 20, "ftol": 1e-6},
        )
        if np.all(np.isfinite(result.x)):
            self._variables = result.x
