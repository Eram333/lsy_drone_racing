"""Check alignment-triggered reference timing and unchanged physical race steps."""

import unittest
from types import SimpleNamespace

import numpy as np

from lsy_drone_racing.control.level2_mpc_guarded import Level2MPCGuarded


def observation(direction: int = 1) -> dict:
    """Return a centered approach to a forward or reverse gate."""
    return {
        "pos": np.array([-0.4 * direction, 0.0, 0.7]),
        "vel": np.array([1.5 * direction, 0.0, 0.0]),
        "n_gates_passed": 0,
        "gate_sequence": np.array([0]),
        "gate_sequence_direction": np.array([direction]),
        "gates_pos": np.array([[0.0, 0.0, 0.7]]),
        "gates_quat": np.array([[0.0, 0.0, 0.0, 1.0]]),
    }


class TestGuardedMPC(unittest.TestCase):
    """Verify rate selection, derivative scaling, and reference-only timing changes."""

    def test_centered_and_distant_approaches_keep_full_rate(self):
        """Centered approaches in both directions and distant gates need no intervention."""
        controller = Level2MPCGuarded.__new__(Level2MPCGuarded)
        for direction in (-1, 1):
            obs = observation(direction)
            self.assertEqual(controller._target_phase_rate(obs), 1.0)
            obs["pos"] = np.array([-2 * direction, 0.2, 0.7])
            obs["vel"][1] = 1.0
            self.assertEqual(controller._target_phase_rate(obs), 1.0)

    def test_lateral_and_vertical_misalignment_reduce_bounded_rate(self):
        """An approaching frame needs more alignment time, including reverse crossings."""
        controller = Level2MPCGuarded.__new__(Level2MPCGuarded)
        for direction in (-1, 1):
            for axis in (1, 2):
                obs = observation(direction)
                obs["pos"][axis] += 0.18
                obs["vel"][axis] = 0.8
                rate = controller._target_phase_rate(obs)
                self.assertGreaterEqual(rate, controller.MIN_PHASE_RATE)
                self.assertLess(rate, 1.0)

    def test_reference_derivatives_use_phase_rate(self):
        """Predicted velocity/acceleration must match the slower trajectory parameter."""
        controller = Level2MPCGuarded.__new__(Level2MPCGuarded)
        controller._N, controller._prediction_dt, controller._phase_rate = 2, 0.04, 0.6
        captured = []

        def sample(obs: dict, offsets: np.ndarray) -> tuple:
            captured.append(offsets.copy())
            return np.ones((3, 3)), np.full((3, 3), 2.0), np.full((3, 3), 3.0)

        controller._trajectory = SimpleNamespace(sample=sample)
        _, velocity = controller._reference({})
        np.testing.assert_allclose(captured[0], [0, 0.024, 0.048])
        np.testing.assert_allclose(velocity, 1.2)
        np.testing.assert_allclose(controller._reference_acceleration, 1.08)

    def test_frame_lead_triggers_before_centre_crossing(self):
        """A front-clearance allowance must intervene before the centre-only estimate."""
        controller = Level2MPCGuarded.__new__(Level2MPCGuarded)
        obs = observation()
        obs["pos"] = np.array([-0.3, 0.08, 0.7])
        obs["vel"] = np.array([1.5, 0.2, 0.0])
        controller.FRAME_LEAD = 0.0
        centre_rate = controller._target_phase_rate(obs)
        controller.FRAME_LEAD = 0.12
        self.assertLess(controller._target_phase_rate(obs), centre_rate)

    def test_callback_preserves_real_tick_and_scales_only_reference(self):
        """The competition's physical step count must not be reduced by the governor."""
        controller = Level2MPCGuarded.__new__(Level2MPCGuarded)
        controller._dt, controller._phase_rate = 0.02, 0.6
        controller._tick, controller._failures, controller._reference_delay = 0, 0, 0.0
        controller._trajectory = SimpleNamespace(_time=0.0)
        self.assertFalse(controller.step_callback(np.zeros(4), {}, 0.0, False, False, {}))
        self.assertEqual(controller._tick, 1)
        self.assertAlmostEqual(controller._trajectory._time, 0.012)
        self.assertAlmostEqual(controller._reference_delay, 0.008)

    def test_gate_deadline_is_converted_to_physical_time(self):
        """A half-rate reference needs twice as much physical time to its next gate."""
        controller = Level2MPCGuarded.__new__(Level2MPCGuarded)
        controller._trajectory = SimpleNamespace(_variables=np.array([1.0]), _time=0.6)
        controller._phase_rate = 0.5
        self.assertAlmostEqual(controller._gate_remaining_time(0), 0.8)
