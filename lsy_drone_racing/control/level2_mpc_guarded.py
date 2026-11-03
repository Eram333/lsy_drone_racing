"""Fast racing MPC with an observation-based gate-approach reference governor."""

import numpy as np
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control import level2_mpc


class Level2MPCGuarded(level2_mpc.Level2MPC):
    """Adjust reference progress near misaligned gates without changing race timing.

    The alignment estimate is ballistic and heuristic, not a safety certificate.
    It only uses the observed gate geometry, position and velocity.
    """

    TIME_SCALE = 1.4
    ALIGNMENT_ACCEL = 4.0
    MIN_PHASE_RATE = 0.6
    SAFE_OFFSET = 0.07
    APPROACH_DISTANCE = 0.7
    FRAME_LEAD = 0.12

    def __init__(self, obs: dict, info: dict, config: dict):
        """Keep the accepted dynamics/solver and initialize reference-rate tracking."""
        if not 0 < self.MIN_PHASE_RATE <= 1 or self.ALIGNMENT_ACCEL <= 0:
            raise ValueError("Use positive alignment acceleration and a phase rate in (0, 1]")
        super().__init__(obs, info, config)
        self._phase_rate = 1.0
        self._minimum_rate = 1.0
        self._governed_steps = 0
        self._reference_delay = 0.0

    def _target_phase_rate(self, obs: dict) -> float:
        """Estimate the correction needed before crossing the observed gate plane."""
        progress = int(obs["n_gates_passed"])
        if progress >= len(obs["gate_sequence"]):
            return 1.0
        gate = int(obs["gate_sequence"][progress])
        rotation = Rotation.from_quat(obs["gates_quat"][gate]).as_matrix()
        delta = np.asarray(obs["pos"]) - obs["gates_pos"][gate]
        local = rotation.T @ delta
        velocity = rotation.T @ np.asarray(obs["vel"])
        direction = int(obs["gate_sequence_direction"][progress])
        distance, closing_speed = -direction * local[0], direction * velocity[0]
        if not 0 < distance < self.APPROACH_DISTANCE or closing_speed <= 0.2:
            return 1.0
        if np.linalg.norm(delta) > 0.9:
            return 1.0
        # Frame contact can precede the centre crossing. Reserve this observed
        # approach distance when estimating how much alignment time remains.
        crossing_time = float(np.clip((distance - self.FRAME_LEAD) / closing_speed, 0.08, 0.6))
        predicted_offset = local[1:] + crossing_time * velocity[1:]
        correction = np.maximum(np.abs(predicted_offset) - self.SAFE_OFFSET, 0)
        required_acceleration = 2 * float(np.max(correction)) / crossing_time**2
        return float(
            np.clip(
                np.sqrt(self.ALIGNMENT_ACCEL / max(required_acceleration, self.ALIGNMENT_ACCEL)),
                self.MIN_PHASE_RATE,
                1.0,
            )
        )

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Reduce planned progress smoothly when lateral/vertical alignment is poor."""
        target = self._target_phase_rate(obs)
        self._phase_rate += float(np.clip(target - self._phase_rate, -0.2, 0.08))
        self._minimum_rate = min(self._minimum_rate, self._phase_rate)
        self._governed_steps += int(self._phase_rate < 0.999)
        return super().compute_control(obs, info)

    def _reference(self, obs: dict) -> tuple[np.ndarray, np.ndarray]:
        offsets = np.arange(self._N + 1) * self._prediction_dt * self._phase_rate
        positions, velocities, acceleration = self._trajectory.sample(obs, offsets)
        self._reference_acceleration = acceleration * self._phase_rate**2
        return positions, velocities * self._phase_rate

    def _gate_remaining_time(self, progress: int) -> float:
        """Convert nominal trajectory time to the slowed physical prediction horizon."""
        return super()._gate_remaining_time(progress) / self._phase_rate

    def step_callback(
        self,
        action: np.ndarray,
        obs: dict,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Slow only the internal reference; environment steps and race time stay unchanged."""
        stopped = super().step_callback(action, obs, reward, terminated, truncated, info)
        delay = (1 - self._phase_rate) * self._dt
        self._trajectory._time -= delay
        self._reference_delay += delay
        return stopped

    def diagnostics(self) -> dict:
        """Report how often the reference governor intervened."""
        return {
            "minimum_phase_rate": self._minimum_rate,
            "governed_steps": self._governed_steps,
            "reference_delay_s": self._reference_delay,
        }
