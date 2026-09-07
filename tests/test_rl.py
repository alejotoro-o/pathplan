import pytest
import numpy as np
from typing import Type
from pathplan.core.base_solver import BaseSolver
from pathplan.core.map import GridMap
from pathplan.rl.q_learning import QLearningPlanner
from pathplan.rl.sarsa import SarsaPlanner

RL_PLANNERS = [
    QLearningPlanner,
    SarsaPlanner,
]


@pytest.mark.parametrize("planner_class", RL_PLANNERS)
def test_start_equals_goal(planner_class: Type[BaseSolver], empty_map_20x20):
    """Verifies edge case behavior when target goal coordinates equal start points."""
    planner = planner_class(empty_map_20x20)
    start = (5, 5)
    goal = (5, 5)

    path, _ = planner.plan(start, goal)

    assert path is not None
    assert len(path) == 1
    assert path[0] == start


@pytest.mark.parametrize("planner_class", RL_PLANNERS)
def test_successful_path_empty_map(planner_class: Type[BaseSolver]):
    """Verifies that RL planners find a path in a small empty map."""
    data = np.zeros((5, 5))
    grid_map = GridMap(data)

    planner = planner_class(grid_map, num_episodes=50, max_steps=50)
    start = (0, 0)
    goal = (4, 4)

    path, _ = planner.plan(start, goal)

    assert path is not None
    assert len(path) >= 2
    assert path[0] == start
    assert path[-1] == goal

    for node in path:
        assert not grid_map.is_occupied(node[0], node[1])


@pytest.mark.parametrize("planner_class", RL_PLANNERS)
def test_unreachable_goal(planner_class: Type[BaseSolver], blocked_map_5x5):
    """Verifies that RL planners return None when a path is physically blocked."""
    planner = planner_class(blocked_map_5x5, num_episodes=50, max_steps=50)
    start = (0, 0)
    goal = (4, 4)

    path, _ = planner.plan(start, goal)
    assert path is None
