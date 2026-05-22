import pytest
from typing import Type
from pathplan.core.base_solver import BaseSolver
from pathplan.classic.a_star import AStarPlanner
from pathplan.classic.rrt import RRTPlanner
from pathplan.classic.rrt_star import RRTStarPlanner
from pathplan.classic.theta_star import ThetaStarPlanner
from pathplan.classic.dijkstra import DijkstraPlanner
from pathplan.classic.bidirectional_a_star import BidirectionalAStarPlanner
from pathplan.classic.jps import JPSPlanner
from pathplan.classic.d_star_lite import DStarLitePlanner
from pathplan.classic.prm import PRMPlanner
from pathplan.classic.informed_rrt_star import InformedRRTStarPlanner
from pathplan.classic.apf import APFPlanner
from pathplan.classic.bug import Bug2Planner

# Tuple array managing solvers slated for uniform black-box interface testing
CLASSIC_PLANNERS = [
    AStarPlanner, 
    RRTPlanner, 
    RRTStarPlanner, 
    ThetaStarPlanner,
    DijkstraPlanner,
    BidirectionalAStarPlanner,
    JPSPlanner,
    DStarLitePlanner,
    PRMPlanner,
    InformedRRTStarPlanner,
    APFPlanner,
    Bug2Planner
]

@pytest.mark.parametrize("planner_class", CLASSIC_PLANNERS)
def test_successful_path_generation(planner_class: Type[BaseSolver], wall_map_20x20):
    """Verifies that classic planners find a valid path through open environment corridors."""
    planner = planner_class(wall_map_20x20)
    start = (0, 0)
    goal = (19, 19)
    
    path, explored = planner.plan(start, goal)
    
    # Assertions
    assert path is not None, f"{planner_class.__name__} failed to discover a valid solution path."
    assert len(path) > 0
    assert path[0] == start, "Generated coordinates do not initiate from start coordinates."
    assert path[-1] == goal, "Generated coordinates fail to reach target goal coordinates."
    
    # Verify path elements do not collide with obstacles
    for node in path:
        # Cast coordinate positions to integer elements to check the Map Matrix bounds safely
        coord_idx = (int(round(node[0])), int(round(node[1])))
        assert not wall_map_20x20.is_occupied(*coord_idx), f"Path intersects obstacle at position: {node}"


@pytest.mark.parametrize("planner_class", CLASSIC_PLANNERS)
def test_unreachable_goal(planner_class: Type[BaseSolver], blocked_map_5x5):
    """Verifies solvers cleanly return None when an impossible boundary is encountered."""
    planner = planner_class(blocked_map_5x5)
    start = (0, 0)
    goal = (4, 4)
    
    path, _ = planner.plan(start, goal)
    assert path is None, f"{planner_class.__name__} should return None for physically blocked goals."


@pytest.mark.parametrize("planner_class", CLASSIC_PLANNERS)
def test_start_equals_goal(planner_class: Type[BaseSolver], empty_map_20x20):
    """Verifies edge case behavior when target goal coordinates equal start points."""
    planner = planner_class(empty_map_20x20)
    start = (5, 5)
    goal = (5, 5)
    
    path, _ = planner.plan(start, goal)
    
    assert path is not None
    assert len(path) == 1
    assert path[0] == start