"""Bounded prediction relaxation must never soften actuator limits or gate rules."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from crazyflow.dynamics import Dynamics, load_params
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.attitude_mpc import create_ocp_solver
from lsy_drone_racing.control.level2_mpc import Level2MPC


class TestSoftGateCorridor(unittest.TestCase):
    """Check signed geometry, effective relaxation caps and unchanged input bounds."""

    def setUp(self):
        """Construct one observed reverse passage without starting a solver."""
        self.controller = c = Level2MPC.__new__(Level2MPC)
        c.GATE_SLACK_CAPS = (0.45, 0.05, 0.05, 1.5)
        c._N, c._prediction_dt, c._nx = 20, 0.04, 12
        c._gate_constraint_stage = None
        c._trajectory = SimpleNamespace(_variables=np.ones(7), _time=0.5)
        c._solver = Mock()
        self.obs = {
            "n_gates_passed": 0,
            "gate_sequence": np.array([0]),
            "gate_sequence_direction": np.array([-1]),
            "gates_pos": np.array([[0, 0, 1.0]]),
            "gates_quat": Rotation.from_euler("xyz", [[0.02, 0.1, 0.8]]).as_quat(),
            "pos": np.array([0.1, 0, 1.0]),
        }

    def values(self) -> dict[str, np.ndarray]:
        """Collect the last bound/matrix sent to the recording solver."""
        return {
            call.args[1]: call.args[2]
            for call in self.controller._solver.constraints_set.call_args_list
        }

    def test_duplicate_outer_rows_have_no_additional_slacks(self):
        """Validate acados dimensions and hard actuator limits without native build."""
        parameters = load_params(Dynamics.so_rpy, "cf21B_500")
        module = "lsy_drone_racing.control.attitude_mpc"
        with patch(module + ".AcadosOcpSolver"), patch(module + "._built_solvers", set()):
            _, ocp = create_ocp_solver(0.8, 20, parameters, racing=True, bounded_gate_slack=True)
        ocp.make_consistent()
        self.assertEqual(ocp.constraints.C.shape, (8, 12))
        np.testing.assert_array_equal(ocp.constraints.idxsg, [0, 1, 2, 3])
        self.assertEqual(len(ocp.constraints.idxsbu), 0)
        np.testing.assert_array_equal(ocp.constraints.lbu[:3], [-1, -1, -0.5])
        np.testing.assert_array_equal(ocp.constraints.ubu[:3], [1, 1, 0.5])
        self.assertEqual(ocp.constraints.ubu[3], 4 * parameters["thrust_max"])
        self.assertEqual(ocp.dims.ng, 8)
        self.assertEqual(ocp.dims.nsg, 4)

    def test_signed_gate_outer_bounds_cap_relaxation(self):
        """Hard duplicate rows bound both directions for a rotated reverse passage."""
        self.controller._set_gate_corridor(self.obs)
        values = self.values()
        np.testing.assert_array_equal(values["C"][:4], values["C"][4:])
        np.testing.assert_allclose(
            values["lg"][:4] - values["lg"][4:], self.controller.GATE_SLACK_CAPS
        )
        np.testing.assert_allclose(
            values["ug"][4:] - values["ug"][:4], self.controller.GATE_SLACK_CAPS
        )
        normal = -Rotation.from_quat(self.obs["gates_quat"][0]).as_matrix()[:, 0]
        np.testing.assert_allclose(values["C"][0, :3], normal)

    def test_deactivation_resets_all_eight_rows(self):
        """An old crossing stage must not constrain the next prediction."""
        c = self.controller
        c._set_gate_corridor(self.obs)
        previous = c._gate_constraint_stage
        c._solver.reset_mock()
        c._trajectory._time = 0
        c._set_gate_corridor(self.obs)
        calls = c._solver.constraints_set.call_args_list
        self.assertIsNone(c._gate_constraint_stage)
        self.assertEqual(calls[0].args[0], previous)
        self.assertEqual(calls[0].args[2].shape, (8,))
        self.assertTrue(np.all(calls[0].args[2] == -1e4))

    def test_original_four_row_mode_is_available(self):
        """The unmodified controller profile remains selectable."""
        self.controller.GATE_SLACK_CAPS = None
        self.controller._set_gate_corridor(self.obs)
        self.assertEqual(self.values()["C"].shape, (4, 12))

    def test_approach_stage_and_its_reset_use_same_hard_caps(self):
        """Both active alignment stages are bounded and subsequently deactivated."""
        c = self.controller
        c.GATE_APPROACH_STEPS = 4
        c._set_gate_corridor(self.obs)
        self.assertEqual(len(c._gate_constraint_stages), 2)
        values = self.values()
        np.testing.assert_allclose(values["lg"][:4] - values["lg"][4:], c.GATE_SLACK_CAPS)
        np.testing.assert_allclose(values["ug"][4:] - values["ug"][:4], c.GATE_SLACK_CAPS)
        c._solver.reset_mock()
        c._trajectory._time = 0
        c._set_gate_corridor(self.obs)
        self.assertEqual(len(c._solver.constraints_set.call_args_list), 4)
        self.assertEqual(c._gate_constraint_stages, [])

    def test_excessive_or_nonfinite_relaxation_rejected(self):
        """Configuration cannot silently expand beyond the permitted relaxation."""
        for caps in (
            (0.45, 0.06, 0.05, 1.5),
            (0.45, 0.05, 0.05, 2.1),
            (np.nan, 0, 0, 0),
            (-0.1, 0, 0, 0),
            (0.1,),
        ):
            with self.subTest(caps=caps), self.assertRaises(ValueError):
                self.controller.GATE_SLACK_CAPS = caps
                self.controller._set_gate_corridor(self.obs)


if __name__ == "__main__":
    unittest.main()
