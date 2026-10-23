"""Continuity, geometry and signed-crossing checks for racing MPC references."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from lsy_drone_racing.control.racing_trajectory import (
    RaceTrajectory,
    frame_clearance,
    quintic_coefficients,
    sample_segments,
)


class TestRacingTrajectory(unittest.TestCase):
    """Validate references independently of the simulator and trajectory optimizer."""

    def test_plan_cache_includes_initial_state(self):
        """Reuse plans only when both geometry and initial position and velocity match."""
        planner = RaceTrajectory.__new__(RaceTrajectory)
        planner._start = np.array([-1.5, 0.75, 0.01])
        planner._start_velocity = np.zeros(3)
        planner._gates = np.array([[0.5, 0.25, 0.7]])
        planner._rotations = np.eye(3)[None]
        planner._obstacles = np.empty((0, 3))
        planner._sequence = np.array([0])
        planner._directions = np.array([1])
        planner._variables = np.r_[1.0, 1.0, np.zeros(5)]
        result = SimpleNamespace(x=planner._variables.copy(), fun=0.0, success=True)
        module = "lsy_drone_racing.control.racing_trajectory"
        with patch.dict(f"{module}._plan_cache", clear=True), patch(
            f"{module}.minimize", return_value=result
        ) as optimize:
            planner._optimize()
            planner._optimize()
            self.assertEqual(optimize.call_count, 1)
            planner._start[0] += 0.1
            planner._optimize()
            self.assertEqual(optimize.call_count, 2)
            planner._start_velocity[1] += 0.1
            planner._optimize()
            self.assertEqual(optimize.call_count, 3)

    def test_endpoint_derivatives(self):
        """Quintics must honor position, velocity and acceleration in physical time."""
        points = np.array([[0, 0, 0.2], [1, 0.5, 0.7], [0, -0.5, 1.2]])
        velocity = np.array([[0, 0, 0], [1, -1, 0.2], [-0.5, 0.7, 0.1]])
        acceleration = np.array([[0, 0, 1], [-2, 1, 0.5], [0.5, -1, -0.2]])
        durations = np.array([0.8, 1.3])
        coefficients = quintic_coefficients(points, velocity, acceleration, durations)
        for derivative, expected in enumerate((points, velocity, acceleration)):
            values = sample_segments(coefficients, durations, np.array([0, 1]), derivative)
            np.testing.assert_allclose(values[:, 0], expected[:-1], atol=1e-11)
            np.testing.assert_allclose(values[:, 1], expected[1:], atol=1e-11)
            np.testing.assert_allclose(values[0, 1], values[1, 0], atol=1e-11)

    def test_gate_opening_and_bars(self):
        """The gate opening must be clear while each solid bar remains an obstacle."""
        positions = np.array([[0, 0, 0], [0, 0.28, 0], [0, 0, 0.28], [0.3, 0.28, 0]])
        distances = frame_clearance(positions, np.zeros((1, 3)), np.eye(3)[None])
        self.assertAlmostEqual(distances[0], 0.2)
        self.assertLess(distances[1], 0)
        self.assertLess(distances[2], 0)
        self.assertGreater(distances[3], 0.25)

    def test_signed_crossing_with_tangential_velocity(self):
        """Crossing-angle optimization must preserve the requested passing direction."""
        planner = RaceTrajectory.__new__(RaceTrajectory)
        planner._sequence = np.array([0, 0])
        planner._directions = np.array([1, -1])
        planner._start = np.array([-1, 0, 0.1])
        planner._geometry(
            {
                "gates_pos": np.array([[0, 0, 0.7]]),
                "gates_quat": np.array([[0, 0, np.sin(0.4), np.cos(0.4)]]),
                "obstacles_pos": np.empty((0, 3)),
            }
        )
        variables = np.r_[[1, 1], [1.2, 1.8], np.zeros(6), [0.4, 0.2, -0.3, 0.1]]
        velocity = planner._gate_velocities(variables)
        np.testing.assert_allclose(np.sum(velocity * planner._normals, axis=1), [1.2, 1.8])

    def test_reference_continues_through_final_gate(self):
        """An exhausted reference must cross the final plane rather than hover just before it."""
        planner = RaceTrajectory.__new__(RaceTrajectory)
        planner._sequence = np.array([0])
        planner._normals = np.array([[1, 0, 0]])
        planner._rotations = np.eye(3)[None]
        planner._points = np.array([[0, 0, 0.7], [1, 0, 0.7]])
        planner._variables = np.r_[1, 1, np.zeros(5)]
        planner._coefficients = quintic_coefficients(
            planner._points, np.array([[0, 0, 0], [1, 0, 0]]), np.zeros((2, 3)), np.array([1])
        )
        positions = planner._reference(np.array([1, 1.1, 1.2]))
        self.assertGreater(positions[1, 0], positions[0, 0])
        self.assertGreater(positions[2, 0], positions[1, 0])


if __name__ == "__main__":
    unittest.main()
