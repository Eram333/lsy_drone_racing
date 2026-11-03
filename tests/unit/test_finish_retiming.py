"""Model-screened final-leg changes must preserve the preceding race reference."""

import unittest

import numpy as np
from crazyflow.dynamics import Dynamics, load_params

from lsy_drone_racing.control.racing_trajectory import RaceTrajectory, sample_segments


class TestFinishRetiming(unittest.TestCase):
    """Exercise actual polynomial and feasibility calculations without a simulator."""

    def setUp(self):
        """Build two straight passages using the actual identified model parameters."""
        self.planner = p = RaceTrajectory.__new__(RaceTrajectory)
        p._start, p._start_velocity = np.array([-1.0, 0, 0.7]), np.array([0.5, 0, 0])
        p._sequence, p._directions = np.array([0, 1]), np.ones(2)
        self.obs = {
            "gates_pos": np.array([[0, 0, 0.7], [2, 0, 0.7]]),
            "gates_quat": np.array([[0, 0, 0, 1], [0, 0, 0, 1]]),
            "obstacles_pos": np.empty((0, 3)),
            "n_gates_passed": 0,
        }
        p._geometry(self.obs)
        p._geometry_key = p._key()
        p._variables = np.r_[[2.0, 2.0], [1.0, 1.0], np.zeros(10)]
        p._coefficients = p._coefficients_for(p._variables)
        p._dynamics = load_params(Dynamics.so_rpy, "cf21B_500")
        p._finish_time_factor, p._finish_factor_used = 0.75, 1.0
        p._last_progress, p._time = 0, 0.0

    def test_retiming_preserves_prefix_and_c2_join(self):
        """A feasible shorter finish must leave the entire preceding leg unchanged."""
        p = self.planner
        original = p._coefficients.copy()
        p._retime_finish()
        self.assertLess(p._finish_factor_used, 1)
        np.testing.assert_array_equal(p._coefficients[:-1], original[:-1])
        for derivative in range(3):
            before = sample_segments(original, np.array([2.0, 2.0]), np.array([0, 1]), derivative)
            after = sample_segments(p._coefficients, p._variables[:2], np.array([0, 1]), derivative)
            np.testing.assert_allclose(after[1, 0], before[1, 0], atol=1e-10)
            np.testing.assert_allclose(after[0, 1], after[1, 0], atol=1e-10)
        self.assertTrue(p._finish_feasible(p._coefficients[-1:], p._variables[1:2]))

    def test_no_change_before_confirmed_passage_or_after_join(self):
        """Neither previewing a leg nor a late crossing may splice a new reference."""
        p = self.planner
        variables = p._variables.copy()
        p._time = 1.9
        p.sample(self.obs, np.array([0.0, 0.8]))
        np.testing.assert_array_equal(p._variables, variables)
        self.obs["n_gates_passed"] = 1
        p._time = 2.1
        p.sample(self.obs, np.array([0.0, 0.8]))
        np.testing.assert_array_equal(p._variables, variables)

    def test_confirmed_join_activates_without_position_or_derivative_jump(self):
        """A confirmed crossing may switch timing only at the shared C2 boundary."""
        p = self.planner
        p._time = 2.0
        expected = [p._reference(np.array([2.0]), d) for d in range(3)]
        self.obs["n_gates_passed"] = 1
        actual = p.sample(self.obs, np.array([0.0]))
        self.assertLess(p._finish_factor_used, 1)
        for old, new in zip(expected, actual):
            np.testing.assert_allclose(old, new, atol=1e-10)

    def test_infeasible_model_or_obstacle_keeps_original_plan(self):
        """Reject insufficient actuator authority and routes through an observed pole."""
        p = self.planner
        original = p._variables.copy()
        p._dynamics = dict(p._dynamics, thrust_max=0.001)
        p._retime_finish()
        np.testing.assert_array_equal(p._variables, original)
        self.assertEqual(p._finish_factor_used, 1)
        p._dynamics = load_params(Dynamics.so_rpy, "cf21B_500")
        p._obstacles = np.array([[1, 0, 1.5]])
        p._retime_finish()
        np.testing.assert_array_equal(p._variables, original)

    def test_rotational_authority_and_nonfinite_demands_are_rejected(self):
        """A translationally feasible curve still needs finite, realizable attitude inputs."""
        p = self.planner
        p._retime_finish()
        coefficients, duration = p._coefficients[-1:], p._variables[1:2]
        self.assertTrue(p._finish_feasible(coefficients, duration))
        p._dynamics = dict(p._dynamics)
        p._dynamics["cmd_rpy_coef"] = np.asarray(p._dynamics["cmd_rpy_coef"]) * 1e-3
        self.assertFalse(p._finish_feasible(coefficients, duration))
        self.assertFalse(p._finish_feasible(coefficients * np.nan, duration))

    def test_reverse_final_crossing_retains_positive_signed_velocity(self):
        """A reversed gate orientation/direction must not reverse the requested crossing."""
        p = self.planner
        self.obs["gates_quat"][1] = [0, 0, 1, 0]
        p._directions[1] = -1
        p._geometry(self.obs)
        p._coefficients = p._coefficients_for(p._variables)
        p._retime_finish()
        self.assertLess(p._finish_factor_used, 1)
        self.assertGreater(p._gate_velocities(p._variables)[-1] @ p._normals[-1], 0)

    def test_inverse_rotational_command_matches_identified_dynamics(self):
        """Verify the screen's inverse-input convention against the actual CasADi model."""
        import casadi as ca

        from lsy_drone_racing.control.attitude_mpc import create_acados_model

        parameters = self.planner._dynamics
        model = create_acados_model(parameters)
        dynamics = ca.Function("finish_dynamics", [model.x, model.u], [model.f_expl_expr])
        state = np.r_[np.zeros(3), [0.15, -0.2, 0.05], np.zeros(3), [0.4, -0.3, 0.2]]
        desired = np.array([-0.2, 0.5, -0.1])
        command = (
            desired
            - parameters["rpy_coef"] * state[3:6]
            - parameters["rpy_rates_coef"] * state[9:12]
        ) / parameters["cmd_rpy_coef"]
        derivative = np.asarray(dynamics(state, np.r_[command, 0.3])).ravel()
        np.testing.assert_allclose(derivative[9:12], desired, atol=1e-11)


if __name__ == "__main__":
    unittest.main()
