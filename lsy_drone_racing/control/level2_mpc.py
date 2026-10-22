"""Observation-based gate planning with nonlinear attitude MPC for level 2.

The racing planner builds a continuous quintic trajectory through signed gate
crossings, updates it from public sensor observations, and supplies position,
velocity and attitude references to the acados MPC. Soft prediction constraints
encourage crossing the current gate opening. Sampled obstacle clearance remains
a planning heuristic; randomized races can still fail. The conservative
visibility-graph planner is available with RACING=False.
"""

from __future__ import annotations

import heapq
import logging
from typing import TYPE_CHECKING

import numpy as np
from crazyflow.dynamics import Dynamics, load_params
from crazyflow.dynamics.utils.rotation import ang_vel2rpy_rates
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.attitude_mpc import create_ocp_solver

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def segment_clear(a: NDArray, b: NDArray, centres: NDArray, radii: NDArray) -> bool:
    """Check a horizontal line segment against inflated circular obstacles."""
    delta = b - a
    t = np.clip((centres - a) @ delta / max(float(delta @ delta), 1e-12), 0, 1)
    distance = np.linalg.norm(centres - a - t[:, None] * delta, axis=1)
    return bool(np.all(distance >= radii - 1e-8))


def transit_path(start: NDArray, goal: NDArray, centres: NDArray, radii: NDArray) -> NDArray:
    """Find a short collision-free polyline through a circular visibility graph."""
    if segment_clear(start[:2], goal[:2], centres, radii):
        return np.array([start, goal])
    angles = np.arange(16) * (2 * np.pi / 16)
    ring = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    candidates = (centres[:, None] + (radii[:, None, None] + 0.04) * ring).reshape(-1, 2)
    valid = np.all(np.linalg.norm(candidates[:, None] - centres, axis=2) >= radii[None], axis=1)
    valid &= (np.abs(candidates[:, 0]) < 2.35) & (np.abs(candidates[:, 1]) < 1.35)
    nodes = np.vstack([start[:2], goal[:2], candidates[valid]])
    distances = np.full(len(nodes), np.inf)
    distances[0] = 0
    previous = np.full(len(nodes), -1, dtype=int)
    queue = [(0.0, 0)]
    while queue:
        cost, index = heapq.heappop(queue)
        if index == 1:
            break
        if cost > distances[index]:
            continue
        delta = nodes - nodes[index]
        length = np.linalg.norm(delta, axis=1)
        offsets = centres - nodes[index]
        fraction = np.clip(offsets @ delta.T / np.maximum(length**2, 1e-12), 0, 1)
        separation = offsets[:, None] - fraction[:, :, None] * delta[None]
        clear = np.all(np.linalg.norm(separation, axis=2) >= radii[:, None] - 1e-8, axis=0)
        for other in np.flatnonzero(clear & (cost + length < distances)):
            distances[other] = cost + length[other]
            previous[other] = index
            heapq.heappush(queue, (distances[other], int(other)))
    if not np.isfinite(distances[1]):
        # Hold rather than deliberately fly through an obstacle.
        return np.array([start, start])
    indices = [1]
    while indices[-1] != 0:
        indices.append(previous[indices[-1]])
    xy = nodes[indices[::-1]]
    lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    z = start[2] + (goal[2] - start[2]) * lengths / max(lengths[-1], 1e-9)
    return np.column_stack([xy, z])


def sample_path(path: NDArray, times: NDArray, speed: float) -> tuple[NDArray, NDArray]:
    """Sample a polyline at bounded speed, braking near its endpoint."""
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.r_[0, np.cumsum(lengths)]
    speed = min(speed, np.sqrt(max(0.0, 1.8 * arc[-1])))
    distance = np.minimum(times * speed, arc[-1])
    pos = np.column_stack([np.interp(distance, arc, path[:, axis]) for axis in range(3)])
    indices = np.minimum(np.searchsorted(arc[1:], distance, side="right"), len(lengths) - 1)
    vel = np.diff(path, axis=0)[indices] / np.maximum(lengths[indices, None], 1e-9) * speed
    vel[distance >= arc[-1] - 1e-6] = 0
    return pos, vel


class Level2MPC(Controller):
    """Replan gate approaches online and track them with an acados nonlinear MPC."""

    RACING = True
    TIME_SCALE = 1.8

    def __init__(self, obs: dict, info: dict, config: dict):
        """Build the dynamics solver and initialize race progress."""
        super().__init__(obs, info, config)
        if config.env.control_mode != "attitude":
            raise ValueError("level2_mpc.py requires env.control_mode = 'attitude'")
        self._dt = 1 / config.env.freq
        self._N = 20 if self.RACING else 25
        self._prediction_dt = 0.04
        self._params = load_params(Dynamics.so_rpy, config.sim.drone)
        self._solver, self._ocp = create_ocp_solver(
            self._N * self._prediction_dt, self._N, self._params, racing=self.RACING
        )
        self._hover = float(self._params["mass"] * -self._params["gravity_vec"][-1])
        self._nx = self._ocp.model.x.rows()
        self._thrust_bias = 0.0
        self._gate = -1
        self._crossing = False
        self._tick = 0
        self._failures = 0
        self._path = None
        self._geometry = None
        self._exit = None
        self._previous_gate_geometry = None
        self._gate_constraint_stage = None
        self._trajectory = None
        if self.RACING:
            from lsy_drone_racing.control.racing_trajectory import RaceTrajectory

            self._trajectory = RaceTrajectory(obs, time_scale=self.TIME_SCALE)

    def _reference(self, obs: dict) -> tuple[NDArray, NDArray]:
        """Plan from available sensor observations without reading simulator state."""
        if getattr(self, "_trajectory", None) is not None:
            times = np.arange(self._N + 1) * self._prediction_dt
            positions, velocities, self._reference_acceleration = self._trajectory.sample(
                obs, times
            )
            return positions, velocities
        pos = np.asarray(obs["pos"], dtype=float)
        progress = int(obs["n_gates_passed"])
        if progress >= len(obs["gate_sequence"]):
            return np.tile(pos, (self._N + 1, 1)), np.zeros((self._N + 1, 3))
        if progress != self._gate:
            if self._previous_gate_geometry is not None:
                self._exit = self._previous_gate_geometry
            self._gate = progress
            self._crossing = False
            self._path = None
            logger.info("Planning gate %d/%d", progress + 1, len(obs["gate_sequence"]))
        if self._exit is not None:
            exit_centre, exit_normal = self._exit
            if (pos - exit_centre) @ exit_normal < 0.25 or np.asarray(
                obs["vel"]
            ) @ exit_normal > 0.3:
                # Clear the previous frame before turning or changing altitude.
                path = np.array([pos, exit_centre + 0.3 * exit_normal])
                times = np.arange(self._N + 1) * self._prediction_dt
                return sample_path(path, times, 0.3)
            self._exit = None
        gate_id = int(obs["gate_sequence"][progress])
        centre = np.asarray(obs["gates_pos"][gate_id], dtype=float)
        rotation = Rotation.from_quat(obs["gates_quat"][gate_id])
        direction = int(obs["gate_sequence_direction"][progress])
        normal = rotation.apply([direction, 0, 0])
        self._previous_gate_geometry = (centre.copy(), normal.copy())
        approach = centre - 0.45 * normal
        centres = list(np.asarray(obs["obstacles_pos"])[:, :2])
        radii = [0.24] * len(centres)
        # Approximate complete frames as inflated horizontal segments. The target
        # gate keeps its opening so a nearby obstacle cannot force a long approach
        # that would run through that obstacle on the way to the gate centre.
        for index, (gate_pos, gate_quat) in enumerate(zip(obs["gates_pos"], obs["gates_quat"])):
            offsets = (-0.28, 0.28) if index == gate_id else (-0.32, -0.16, 0, 0.16, 0.32)
            for sideways in offsets:
                point = gate_pos + Rotation.from_quat(gate_quat).apply([0, sideways, 0])
                centres.append(point[:2])
                radii.append(0.15 if index == gate_id else 0.2)
        centres, radii = np.asarray(centres), np.asarray(radii)
        for offset in (0.45, 0.35, 0.3, 0.25, 0.18, 0.12, 0.08, 0.6, 0.75):
            candidate = centre - offset * normal
            crossing_clear = segment_clear(candidate[:2], centre[:2], centres, radii)
            if crossing_clear and np.all(
                np.linalg.norm(centres - candidate[:2], axis=1) > radii + 0.01
            ):
                approach = candidate
                break
        local = rotation.apply(pos - centre, inverse=True)
        if direction * local[0] < -0.04 and np.linalg.norm(pos - approach) < 0.12:
            self._crossing = True
        # A missed crossing must return to the approach side before retrying.
        if self._crossing and direction * local[0] > 0.4:
            self._crossing = False
        target = centre + 0.4 * normal if self._crossing else approach
        if self._tick * self._dt < 1.5:
            target = np.array([pos[0], pos[1], 0.45])
            path = np.array([pos, target])
        elif self._crossing:
            path = (
                np.array([pos, centre, target])
                if direction * local[0] < 0
                else np.array([pos, target])
            )
        else:
            # Allow recovery when a newly revealed object overlaps the current clearance disc.
            radii = np.minimum(
                radii, np.maximum(0.03, np.linalg.norm(centres - pos[:2], axis=1) - 0.01)
            )
            geometry = np.r_[centres.ravel(), target]
            if self._path is None or not np.array_equal(geometry, self._geometry):
                self._path = transit_path(pos, target, centres, radii)
                self._geometry = geometry.copy()
            if len(self._path) > 2:
                segments = np.diff(self._path[:, :2], axis=0)
                fractions = np.clip(
                    np.sum((pos[:2] - self._path[:-1, :2]) * segments, axis=1)
                    / np.maximum(np.sum(segments**2, axis=1), 1e-9),
                    0,
                    1,
                )
                closest = self._path[:-1, :2] + fractions[:, None] * segments
                nearest = int(np.argmin(np.linalg.norm(closest - pos[:2], axis=1)))
                self._path = self._path[nearest:]
            while len(self._path) > 2 and np.linalg.norm(pos[:2] - self._path[1, :2]) < 0.07:
                self._path = self._path[1:]
            # Track only the next clear segment, preventing shortcuts at corners.
            path = np.array([pos, self._path[1]])
        times = np.arange(self._N + 1) * self._prediction_dt
        return sample_path(path, times, 0.55 if self._crossing else 0.9)

    def compute_control(self, obs: dict, info: dict | None = None) -> NDArray:
        """Solve the finite-horizon tracking problem and apply its first input."""
        rpy = Rotation.from_quat(obs["quat"]).as_euler("xyz")
        rates = np.asarray(ang_vel2rpy_rates(obs["quat"], obs["ang_vel"]), dtype=float)
        state = np.concatenate([obs["pos"], rpy, obs["vel"], rates]).astype(float)
        self._solver.set(0, "lbx", state)
        self._solver.set(0, "ubx", state)
        positions, velocities = self._reference(obs)
        if self.RACING:
            self._set_gate_corridor(obs)
        reference_rpy = np.zeros((self._N + 1, 3))
        reference_rates = np.zeros_like(reference_rpy)
        reference_thrust = np.full(self._N + 1, self._hover)
        if self.RACING:
            forces = self._reference_acceleration + [0, 0, 9.81]
            reference_thrust = np.linalg.norm(forces, axis=1) * self._params["mass"]
            z_axis = forces / np.maximum(np.linalg.norm(forces, axis=1, keepdims=True), 1e-9)
            y_axis = np.cross(z_axis, np.array([1, 0, 0]))
            y_axis /= np.maximum(np.linalg.norm(y_axis, axis=1, keepdims=True), 1e-9)
            x_axis = np.cross(y_axis, z_axis)
            reference_rpy = Rotation.from_matrix(
                np.stack([x_axis, y_axis, z_axis], axis=-1)
            ).as_euler("xyz")
            reference_rates = np.gradient(reference_rpy, self._prediction_dt, axis=0)
        # Slow integral trim compensates for randomized mass and steady vertical disturbance.
        self._thrust_bias = float(
            np.clip(
                self._thrust_bias + 0.025 * (positions[-1, 2] - state[2]) * self._dt, -0.065, 0.065
            )
        )
        for stage in range(self._N):
            reference = np.zeros(self._nx + 4)
            reference[:3] = positions[stage]
            reference[3:6] = reference_rpy[stage]
            reference[6:9] = velocities[stage]
            reference[9:12] = reference_rates[stage]
            reference[self._nx : self._nx + 3] = reference_rpy[stage]
            reference[self._nx + 3] = reference_thrust[stage] + self._thrust_bias
            self._solver.set(stage, "yref", reference)
        terminal = np.zeros(self._nx)
        terminal[:3] = positions[-1]
        terminal[3:6] = reference_rpy[-1]
        terminal[6:9] = velocities[-1]
        terminal[9:12] = reference_rates[-1]
        self._solver.set(self._N, "yref", terminal)
        status = self._solver.solve()
        action = self._solver.get(0, "u")
        if status not in (0, 2) or not np.all(np.isfinite(action)):
            self._failures += 1
            if self._failures == 1:
                logger.warning("MPC solve failed (status %s); applying hover fallback", status)
            action = np.array([0, 0, 0, self._hover + self._thrust_bias])
        else:
            self._failures = 0
        return np.clip(action, self._ocp.constraints.lbu, self._ocp.constraints.ubu).astype(
            np.float32
        )

    def _set_gate_corridor(self, obs: dict) -> None:
        """Require the prediction to pass through the current observed gate opening."""
        if self._gate_constraint_stage is not None:
            self._solver.constraints_set(self._gate_constraint_stage, "lg", np.full(4, -1e4))
            self._solver.constraints_set(self._gate_constraint_stage, "ug", np.full(4, 1e4))
        progress = int(obs["n_gates_passed"])
        if progress >= len(obs["gate_sequence"]):
            return
        index = int(obs["gate_sequence"][progress])
        centre = np.asarray(obs["gates_pos"][index], dtype=float)
        rotation = Rotation.from_quat(obs["gates_quat"][index]).as_matrix()
        normal = rotation[:, 0] * int(obs["gate_sequence_direction"][progress])
        distance = max(0.0, -(np.asarray(obs["pos"]) - centre) @ normal)
        remaining = float(
            np.sum(self._trajectory._variables[: progress + 1]) - self._trajectory._time
        )
        deadline = max(remaining, 0.2, np.sqrt(2 * distance / 8))
        if deadline > (self._N - 1) * self._prediction_dt:
            self._gate_constraint_stage = None
            return
        stage = int(np.clip(np.ceil(deadline / self._prediction_dt), 1, self._N - 1))
        matrix = np.zeros((4, getattr(self, "_nx", 12)))
        matrix[0, :3] = normal
        matrix[1, :3] = rotation[:, 1]
        matrix[2, :3] = rotation[:, 2]
        matrix[3, 6:9] = normal
        projected = matrix[:3, :3] @ centre
        lower = np.r_[projected + [0.06, -0.09, -0.09], 0.3]
        upper = np.r_[projected + [0.3, 0.09, 0.09], 5.0]
        self._solver.constraints_set(stage, "C", matrix, api="new")
        self._solver.constraints_set(stage, "lg", lower)
        self._solver.constraints_set(stage, "ug", upper)
        self._gate_constraint_stage = stage

    def step_callback(
        self,
        action: NDArray,
        obs: dict,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Continue until the environment ends the race or repeated solver failures occur."""
        self._tick += 1
        if self._trajectory is not None:
            self._trajectory._time += self._dt
        return bool(terminated or truncated or self._failures >= 25)
