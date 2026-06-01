import pytest
import numpy as np
from typing import Type
from pathplan.core.base_solver import BaseSolver
from pathplan.metaheuristic.ga import GAPlanner

@pytest.mark.parametrize("planner_class", [GAPlanner])
def test_ga_start_equals_goal(planner_class: Type[BaseSolver], empty_map_20x20):
    """Verifies edge case behavior when target goal coordinates equal start points."""
    planner = planner_class(empty_map_20x20)
    start = (5.0, 5.0)
    goal = (5.0, 5.0)
    
    path, _ = planner.plan(start, goal)
    
    assert path is not None
    assert len(path) == 1
    assert path[0] == start

@pytest.mark.parametrize("planner_class", [GAPlanner])
def test_ga_successful_path_empty_map(planner_class: Type[GAPlanner], empty_map_20x20):
    """Verifies that GAPlanner finds a path in an empty map."""
    # Using small iterations and population for faster testing
    planner = planner_class(empty_map_20x20, num_waypoints=3, pop_size=20, max_iter=50)
    start = (0.0, 0.0)
    goal = (19.0, 19.0)
    
    path, explored = planner.plan(start, goal)
    
    assert path is not None
    assert len(path) == 5 # start + 3 waypoints + goal
    assert path[0] == start
    assert path[-1] == goal
    
    # Verify path elements do not collide (none in empty map anyway)
    for node in path:
        coord_idx = (int(round(node[0])), int(round(node[1])))
        assert not empty_map_20x20.is_occupied(*coord_idx)

@pytest.mark.parametrize("planner_class", [GAPlanner])
def test_ga_unreachable_goal(planner_class: Type[GAPlanner], blocked_map_5x5):
    """Verifies that GAPlanner returns None when a path is physically blocked."""
    planner = planner_class(blocked_map_5x5, num_waypoints=2, pop_size=20, max_iter=20)
    start = (0.0, 0.0)
    goal = (4.0, 4.0)
    
    path, _ = planner.plan(start, goal)
    assert path is None
