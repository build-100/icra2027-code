"""Hand-constructed successful trajectories, independent of experiment records."""
import copy


def three_robot_success(config):
    """Visit all seven regions and connect three initially separate robots."""
    cfg = copy.deepcopy(config)
    cfg["environment"].update(
        graph_path="configs/simulation/maps/7node/c7", map_width=2.0,
        robot_count=3, start_nodes=[0, 2, 4], t_max=3,
    )
    # These routes are a synthetic regression fixture, not a reported policy.
    routes = [[0, 1, 6, 5], [2, 3, 2, 1], [4, 3, 2, 5]]
    return cfg, routes
