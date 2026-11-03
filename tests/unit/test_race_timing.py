"""Physical boundary conditions for selective racing trajectory timing."""

import unittest

import numpy as np

from lsy_drone_racing.control.racing_trajectory import (
    RaceTrajectory,
    retime_sections,
    sample_segments,
)


class TestRaceTiming(unittest.TestCase):
    """Check smooth passage-order retiming, including signed/repeated crossings."""

    def test_retiming_preserves_gate_positions_and_c2_continuity(self):
        """Faster legs must still meet shared positions, velocities and accelerations."""
        rng = np.random.default_rng(29)
        count = 5
        variables = np.r_[np.full(count, 1.5), np.ones(count), rng.normal(size=5 * count)]
        original = variables.copy()
        updated = retime_sections(variables, (0.9, 1, 1, 0.9), (1, 1, 1, 1.1))
        points = np.array(
            [[-1, 0, 0.1], [0, 0, 0.7], [1, 1, 1.2], [-1, 0, 0.7], [0, -1, 1.2], [1, 1, 1.2]]
        )
        directions = np.array([1, -1, 1, 1, -1])
        velocity = np.vstack(
            [
                np.zeros(3),
                np.column_stack(
                    [
                        updated[count : 2 * count] * directions,
                        updated[5 * count :].reshape(count, 2),
                    ]
                ),
            ]
        )
        acceleration = np.vstack([np.zeros(3), updated[2 * count : 5 * count].reshape(count, 3)])
        planner = RaceTrajectory.__new__(RaceTrajectory)
        planner._sequence = np.array([0, 1, 0, 2, 1])
        planner._points = points
        planner._start_velocity = np.zeros(3)
        planner._normals = np.column_stack([directions, np.zeros((count, 2))])
        planner._rotations = np.tile(np.eye(3), (3, 1, 1))
        coefficients = planner._coefficients_for(updated)
        for derivative, expected in enumerate((points, velocity, acceleration)):
            values = sample_segments(coefficients, updated[:count], np.array([0, 1]), derivative)
            np.testing.assert_allclose(values[:, 0], expected[:-1], atol=1e-10)
            np.testing.assert_allclose(values[:, 1], expected[1:], atol=1e-10)
            np.testing.assert_allclose(values[:-1, 1], values[1:, 0], atol=1e-10)
        self.assertLess(updated[:count].sum(), variables[:count].sum())
        np.testing.assert_allclose(updated[count : 2 * count] * directions, [1, -1, 1, 1.1, -1])
        np.testing.assert_array_equal(
            updated[2 * count : 5 * count], variables[2 * count : 5 * count]
        )
        np.testing.assert_array_equal(variables, original)

    def test_invalid_factors_are_rejected(self):
        """Zero, negative and nonfinite time scales must never reach flight control."""
        variables = np.r_[1.0, 1.0, np.zeros(5)]
        for factor in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(factor=factor):
                with self.assertRaises(ValueError):
                    retime_sections(variables, (factor,))
                with self.assertRaises(ValueError):
                    retime_sections(variables, (), (factor,))


if __name__ == "__main__":
    unittest.main()
