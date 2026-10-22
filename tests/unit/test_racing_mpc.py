"""Signed gate-corridor regression tests."""

import unittest
from types import SimpleNamespace

import numpy as np

from lsy_drone_racing.control.level2_mpc import Level2MPC


class RecordingSolver:
    """Capture online constraints without constructing an acados library."""

    def __init__(self):
        """Initialize the recorded constraint fields."""
        self.fields = {}

    def constraints_set(self, stage: int, field: str, value: np.ndarray, **kwargs: str) -> None:
        """Record independent copies of the controller's constraints."""
        self.fields[stage, field] = value.copy()


class TestRacingMPC(unittest.TestCase):
    """Check the controller's signed gate geometry."""

    def test_reverse_gate_corridor(self):
        """A reverse pass must constrain position and velocity along the negative normal."""
        controller = Level2MPC.__new__(Level2MPC)
        controller._solver = RecordingSolver()
        controller._N, controller._prediction_dt = 20, 0.04
        controller._gate_constraint_stage = None
        controller._trajectory = SimpleNamespace(_variables=np.array([0.4]), _time=0.0)
        obs = {
            "pos": np.array([0.4, 0, 0.7]),
            "n_gates_passed": 0,
            "gate_sequence": np.array([0]),
            "gate_sequence_direction": np.array([-1]),
            "gates_pos": np.array([[0, 0, 0.7]]),
            "gates_quat": np.array([[0, 0, 0, 1]]),
        }
        controller._set_gate_corridor(obs)
        stage = controller._gate_constraint_stage
        matrix = controller._solver.fields[stage, "C"]
        lower = controller._solver.fields[stage, "lg"]
        upper = controller._solver.fields[stage, "ug"]
        state = np.zeros(12)
        state[:3] = [-0.1, 0, 0.7]
        state[6:9] = [-0.6, 0, 0]
        self.assertTrue(np.all(matrix @ state >= lower))
        self.assertTrue(np.all(matrix @ state <= upper))
        state[0] = 0.1
        self.assertLess((matrix @ state)[0], lower[0])


if __name__ == "__main__":
    unittest.main()
