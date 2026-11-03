"""Compare analytic costs/gradients with the original independent NumPy objective."""

import unittest
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.racing_trajectory import RaceTrajectory
from lsy_drone_racing.control.trajectory_gradient import TrajectoryCost


class TestTrajectoryGradient(unittest.TestCase):
    """Check signed geometry, moving starts and repair continuity penalties."""

    def setUp(self):
        """Build a small non-axis-aligned track without running an optimizer."""
        self.planner = planner = RaceTrajectory.__new__(RaceTrajectory)
        planner._start = np.array([-1.5, 0.75, 0.01])
        planner._start_velocity = np.array([0.05, -0.02, 0.01])
        planner._sequence = np.array([0, 1, 0])
        planner._directions = np.array([1, -1, 1])
        self.obs = {
            "gates_pos": np.array([[0.5, 0.25, 0.7], [1.05, 0.75, 1.2]]),
            "gates_quat": Rotation.from_euler(
                "xyz", [[0.02, 0.04, -0.78], [0, 0.02, 2.35]]
            ).as_quat(),
            "obstacles_pos": np.array([[0, 0.75, 1.55], [1, 0.25, 1.55]]),
        }
        planner._geometry(self.obs)
        rng = np.random.default_rng(15)
        planner._variables = np.r_[
            [1.1, 1.3, 1.0], [0.8, 0.7, 0.9], rng.normal(size=9), rng.normal(scale=0.1, size=6)
        ]

    def assert_gradient(
        self,
        evaluate: Callable[[np.ndarray], tuple[float, np.ndarray]],
        objective: Callable[[np.ndarray], float],
        variables: np.ndarray,
    ) -> None:
        """Use central finite differences as an independent derivative check."""
        value, analytic = evaluate(variables)
        self.assertAlmostEqual(value, objective(variables), places=7)
        numerical = np.empty_like(variables)
        for index in range(len(variables)):
            plus, minus = variables.copy(), variables.copy()
            plus[index] += 1e-6
            minus[index] -= 1e-6
            numerical[index] = (objective(plus) - objective(minus)) / 2e-6
        np.testing.assert_allclose(analytic, numerical, rtol=3e-4, atol=3e-3)

    def test_objective_and_gradient_match_numpy(self):
        """Preserve all geometry, force, velocity, floor and jerk penalties."""
        planner = self.planner
        evaluate = TrajectoryCost(planner).bind(planner)
        self.assert_gradient(evaluate, planner._objective, planner._variables)

    def test_repair_penalty_gradient(self):
        """Include the current segment's position/velocity continuity in the derivative."""
        planner = self.planner
        count, segment, tau = 3, 1, 0.35
        original = planner._variables.copy()
        basis = tau ** np.arange(6)
        dbasis = np.r_[0, np.arange(1, 6) * tau ** np.arange(5)] / original[segment]
        old_position, old_velocity = np.array([0.7, 0.4, 0.9]), np.array([0.1, -0.2, 0.3])
        onehot = np.eye(count)[segment]
        anchor = np.r_[onehot, basis, dbasis, old_position, old_velocity]

        def objective(variables: np.ndarray) -> float:
            polynomial = planner._coefficients_for(variables)[segment]
            return (
                planner._objective(variables)
                + 200 * np.sum((basis @ polynomial - old_position) ** 2)
                + 2 * np.sum((dbasis @ polynomial - old_velocity) ** 2)
                + 0.05 * np.sum((variables[2 * count :] - original[2 * count :]) ** 2)
            )

        evaluate = TrajectoryCost(planner).bind(planner, anchor, original)
        variables = original + np.linspace(-0.015, 0.02, len(original))
        self.assert_gradient(evaluate, objective, variables)

    def test_nonzero_start_acceleration_gradient(self):
        """Suffix costs must include their inherited C2 acceleration boundary."""
        self.planner._start_acceleration = np.array([0.8, -0.3, 0.2])
        self.assert_gradient(
            TrajectoryCost(self.planner).bind(self.planner),
            self.planner._objective,
            self.planner._variables,
        )

    def test_sensor_updates_and_empty_obstacle_set(self):
        """Rebind new geometry and handle tracks without poles without stale data."""
        planner = self.planner
        cost = TrajectoryCost(planner)
        self.obs["gates_pos"][1] += [0.1, -0.07, 0.02]
        planner._geometry(self.obs)
        self.assert_gradient(cost.bind(planner), planner._objective, planner._variables)
        self.obs["obstacles_pos"] = np.empty((0, 3))
        planner._geometry(self.obs)
        self.assert_gradient(
            TrajectoryCost(planner).bind(planner), planner._objective, planner._variables
        )

    def test_failed_repair_preserves_existing_variables(self):
        """Reject nonfinite or higher-cost optimizer results instead of corrupting a plan."""
        planner = self.planner
        planner._time = 0.3
        planner._coefficients = planner._coefficients_for(planner._variables)
        planner._gradient_cost = TrajectoryCost(planner)
        position = planner._reference(np.array([planner._time]))[0]
        velocity = planner._reference(np.array([planner._time]), 1)[0]
        original = planner._variables.copy()
        for cost in (np.nan, np.inf, 1e30):
            with self.subTest(cost=cost):
                candidate = SimpleNamespace(x=original + 0.1, fun=cost)
                with patch(
                    "lsy_drone_racing.control.racing_trajectory.minimize", return_value=candidate
                ):
                    planner._repair_plan(position, velocity)
                np.testing.assert_array_equal(planner._variables, original)


if __name__ == "__main__":
    unittest.main()
