"""Faster racing-MPC timing for the online competition submission."""

from lsy_drone_racing.control import level2_mpc


class Level2MPCFast(level2_mpc.Level2MPC):
    """Retain the accepted MPC with the previously tested 1.4 reference time scale."""

    TIME_SCALE = 1.4
