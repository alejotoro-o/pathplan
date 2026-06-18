import pytest
import numpy as np
from typing import Type
from pathplan.core.base_solver import BaseSolver
from pathplan.metaheuristic.ga import GAPlanner
from pathplan.metaheuristic.pso import PSOPlanner
from pathplan.metaheuristic.gwo import GWOPlanner
from pathplan.metaheuristic.woa import WOAPlanner

METAHEURISTIC_PLANNERS = [
    GAPlanner,
    PSOPlanner,
    GWOPlanner,
    WOAPlanner,
]

@pytest.mark.parametrize("planner_class", METAHEURISTIC_PLANNERS)
def test_start_equals_goal(planner_class: Type[BaseSolver], empty_map_20x20):
    """Verifies edge case behavior when target goal coordinates equal start points."""
    planner = planner_class(empty_map_20x20)
    start = (5.0, 5.0)
    goal = (5.0, 5.0)
    
    path, _ = planner.plan(start, goal)
    
    assert path is not None
    assert len(path) == 1
    assert path[0] == start

@pytest.mark.parametrize("planner_class", METAHEURISTIC_PLANNERS)
def test_successful_path_empty_map(planner_class: Type[BaseSolver], empty_map_20x20):
    """Verifies that metaheuristic planners find a path in an empty map."""
    planner = planner_class(empty_map_20x20, num_waypoints=3, pop_size=20, max_iter=50)
    start = (0.0, 0.0)
    goal = (19.0, 19.0)
    
    path, explored = planner.plan(start, goal)
    
    assert path is not None
    assert len(path) == 5 # start + 3 waypoints + goal
    assert path[0] == start
    assert path[-1] == goal
    
    for node in path:
        coord_idx = (int(round(node[0])), int(round(node[1])))
        assert not empty_map_20x20.is_occupied(*coord_idx)

@pytest.mark.parametrize("planner_class", METAHEURISTIC_PLANNERS)
def test_unreachable_goal(planner_class: Type[BaseSolver], blocked_map_5x5):
    """Verifies that metaheuristic planners return None when a path is physically blocked."""
    planner = planner_class(blocked_map_5x5, num_waypoints=2, pop_size=20, max_iter=20)
    start = (0.0, 0.0)
    goal = (4.0, 4.0)
    
    path, _ = planner.plan(start, goal)
    assert path is None
