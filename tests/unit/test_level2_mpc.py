"""Geometry and sensor-update regression tests for the level 2 planner."""

import unittest

import numpy as np

from lsy_drone_racing.control.level2_mpc import Level2MPC, sample_path, segment_clear, transit_path


class TestLevel2Planner(unittest.TestCase):
    """Check route clearance, gate directions and online geometry updates."""

    def test_route_avoids_obstacle(self):
        """A blocked direct segment should produce a clear detour."""
        centres, radii = np.array([[0.0, 0.0]]), np.array([0.3])
        path = transit_path(np.array([-1.0, 0, 0.6]), np.array([1.0, 0, 1.2]), centres, radii)
        self.assertGreater(len(path), 2)
        for start, end in zip(path[:-1], path[1:]):
            self.assertTrue(segment_clear(start[:2], end[:2], centres, radii))
        np.testing.assert_allclose(path[[0, -1], 2], [0.6, 1.2])

    def test_blocked_destination_holds_position(self):
        """An unreachable destination should not produce a collision route."""
        start = np.array([-1.0, 0, 0.6])
        path = transit_path(start, np.array([0.0, 0, 0.6]), np.array([[0, 0]]), np.array([0.3]))
        np.testing.assert_allclose(path, np.array([start, start]))

    def test_stationary_reference_is_finite(self):
        """A hold or exhausted path must have a finite, zero-velocity reference."""
        point = np.array([0.1, 0.2, 0.8])
        positions, velocities = sample_path(np.array([point, point]), np.arange(26) * 0.04, 0.6)
        np.testing.assert_allclose(positions, np.tile(point, (26, 1)))
        np.testing.assert_array_equal(velocities, np.zeros((26, 3)))

    def test_gate_direction_and_sensor_refresh(self):
        """Signed directions and newly revealed centres must update the approach."""
        for direction in (-1, 1):
            with self.subTest(direction=direction):
                controller = Level2MPC.__new__(Level2MPC)
                controller._N, controller._prediction_dt, controller._dt = 25, 0.04, 0.02
                controller._tick, controller._gate = 100, -1
                controller._path = controller._geometry = controller._exit = None
                controller._previous_gate_geometry = None
                obs = {
                    "pos": np.array([-direction * 1.0, 0, 0.7]),
                    "vel": np.zeros(3),
                    "n_gates_passed": 0,
                    "gate_sequence": np.array([0]),
                    "gate_sequence_direction": np.array([direction]),
                    "gates_pos": np.array([[0.0, 0, 0.7]]),
                    "gates_quat": np.array([[0, 0, 0, 1]]),
                    "obstacles_pos": np.empty((0, 3)),
                }
                controller._reference(obs)
                first_goal = controller._path[-1].copy()
                self.assertLess(direction * first_goal[0], 0)
                obs["gates_pos"][0, 1] += 0.1
                controller._reference(obs)
                self.assertAlmostEqual(controller._path[-1, 1] - first_goal[1], 0.1)

    def test_obstacle_behind_gate_allows_short_approach(self):
        """A pole behind a gate must not force a crossing through the pole."""
        controller = Level2MPC.__new__(Level2MPC)
        controller._N, controller._prediction_dt, controller._dt = 25, 0.04, 0.02
        controller._tick, controller._gate = 100, -1
        controller._path = controller._geometry = controller._exit = None
        controller._previous_gate_geometry = None
        obs = {
            "pos": np.array([-1.2, -0.3, 0.8]),
            "vel": np.zeros(3),
            "n_gates_passed": 0,
            "gate_sequence": np.array([0]),
            "gate_sequence_direction": np.array([1]),
            "gates_pos": np.array([[0, -0.75, 1.2]]),
            "gates_quat": np.array([[0, 0, 0, 1]]),
            "obstacles_pos": np.array([[-0.393, -0.808, 1.55]]),
        }
        controller._reference(obs)
        approach = controller._path[-1]
        self.assertGreater(approach[0], -0.18)
        self.assertFalse(np.array_equal(controller._path[0], approach))
        self.assertTrue(
            segment_clear(
                approach[:2], obs["gates_pos"][0, :2], obs["obstacles_pos"][:, :2], np.array([0.24])
            )
        )


if __name__ == "__main__":
    unittest.main()
