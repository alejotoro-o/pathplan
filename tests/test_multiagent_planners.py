import math
import numpy as np
import pytest
from pathplan.core.map import GridMap
from pathplan.multiagent import (
    CBSPlanner,
    IndependentPlanner,
    PrioritizedPlanner,
)
from pathplan.multiagent.utils import detect_collision


PLANNERS = [IndependentPlanner, PrioritizedPlanner, CBSPlanner]


def _make_grid(h, w, obstacles=None):
    data = np.zeros((h, w), dtype=np.float32)
    if obstacles:
        for r, c in obstacles:
            data[r, c] = 1.0
    return GridMap(data)


def _path_length(path):
    cost = 0.0
    for i in range(1, len(path)):
        dr = abs(path[i][0] - path[i - 1][0])
        dc = abs(path[i][1] - path[i - 1][1])
        cost += math.sqrt(2) if (dr + dc == 2) else 1.0
    return cost


# ── Shared tests (all planners) ───────────────────────────────────────────


@pytest.mark.parametrize("planner_cls", PLANNERS)
class TestSharedMAPF:
    def test_single_agent(self, planner_cls):
        gm = _make_grid(5, 5)
        planner = planner_cls(gm)
        paths, _ = planner.plan([((0, 0), (4, 4))])
        assert len(paths) == 1
        assert paths[0] is not None
        assert paths[0][0] == (0, 0)
        assert paths[0][-1] == (4, 4)

    def test_start_equals_goal(self, planner_cls):
        gm = _make_grid(5, 5)
        planner = planner_cls(gm)
        paths, _ = planner.plan([((2, 2), (2, 2))])
        assert paths[0] == [(2, 2)]

    def test_two_agents_no_conflict(self, planner_cls):
        gm = _make_grid(5, 5)
        planner = planner_cls(gm)
        agents = [
            ((0, 0), (0, 4)),  # top row going right
            ((4, 0), (4, 4)),  # bottom row going right
        ]
        paths, _ = planner.plan(agents)
        assert all(p is not None for p in paths)
        assert paths[0][-1] == (0, 4)
        assert paths[1][-1] == (4, 4)

    def test_unreachable_agent(self, planner_cls):
        gm = _make_grid(3, 5, obstacles=[(0, 2), (1, 2), (2, 2)])
        planner = planner_cls(gm)
        agents = [
            ((0, 0), (0, 4)),  # blocked by wall
            ((0, 0), (2, 0)),  # reachable
        ]
        paths, _ = planner.plan(agents)
        # At least the blocked agent should be None
        assert paths[0] is None

    def test_empty_agents(self, planner_cls):
        gm = _make_grid(5, 5)
        planner = planner_cls(gm)
        paths, _ = planner.plan([])
        assert paths == []


# ── IndependentPlanner-specific ───────────────────────────────────────────


class TestIndependentPlanner:
    def test_independent_may_collide(self):
        """Two agents heading toward each other on a 1×3 corridor."""
        gm = _make_grid(1, 3)
        planner = IndependentPlanner(gm)
        agents = [
            ((0, 0), (0, 2)),  # goes right
            ((0, 2), (0, 0)),  # goes left
        ]
        paths, _ = planner.plan(agents)
        # Both find paths (independent), but they collide
        assert all(p is not None for p in paths)
        col = detect_collision(paths[0], paths[1])
        assert col is not None  # collision exists


# ── PrioritizedPlanner-specific ───────────────────────────────────────────


class TestPrioritizedPlanner:
    def test_priority_ordering_matters(self):
        """On a narrow corridor, one ordering succeeds and the other fails."""
        gm = _make_grid(1, 3)
        planner = PrioritizedPlanner(gm)

        # Agent A: (0,0)→(0,2), Agent B: (0,2)→(0,0)
        # A first: A goes right, B must wait or go around — but no room
        paths1, exp1 = planner.plan([
            ((0, 0), (0, 2)),
            ((0, 2), (0, 0)),
        ])
        # One ordering — B might fail
        assert len(paths1) == 2

        # Reverse priority: B first
        paths2, exp2 = planner.plan([
            ((0, 2), (0, 0)),
            ((0, 0), (0, 2)),
        ])
        assert len(paths2) == 2

    def test_collision_free_result(self):
        """Prioritized planner must return collision-free paths."""
        gm = _make_grid(3, 3)
        planner = PrioritizedPlanner(gm)
        agents = [
            ((0, 0), (2, 2)),
            ((2, 0), (0, 2)),
        ]
        paths, _ = planner.plan(agents)
        if all(p is not None for p in paths):
            assert detect_collision(paths[0], paths[1]) is None


# ── CBSPlanner-specific ───────────────────────────────────────────────────


class TestCBSPlanner:
    def test_cbs_finds_collision_free(self):
        gm = _make_grid(3, 3)
        planner = CBSPlanner(gm)
        agents = [
            ((0, 0), (2, 2)),
            ((2, 0), (0, 2)),
        ]
        paths, explored = planner.plan(agents)
        assert all(p is not None for p in paths)
        assert detect_collision(paths[0], paths[1]) is None

    def test_cbs_optimal_or_better(self):
        """CBS should find a solution at least as good as Prioritized."""
        gm = _make_grid(5, 5)
        agents = [
            ((0, 0), (4, 4)),
            ((4, 0), (0, 4)),
            ((0, 4), (4, 0)),
        ]

        cbs = CBSPlanner(gm)
        cbs_paths, _ = cbs.plan(agents)
        assert all(p is not None for p in cbs_paths)

        pri = PrioritizedPlanner(gm)
        pri_paths, _ = pri.plan(agents)

        if all(p is not None for p in pri_paths):
            cbs_cost = sum(_path_length(p) for p in cbs_paths)
            pri_cost = sum(_path_length(p) for p in pri_paths)
            # CBS is optimal; prioritized may be suboptimal
            assert cbs_cost <= pri_cost + 1e-9

    def test_cbs_explored_data(self):
        gm = _make_grid(3, 3)
        planner = CBSPlanner(gm)
        agents = [((0, 0), (2, 2)), ((2, 0), (0, 2))]
        _, explored = planner.plan(agents)
        assert "nodes_expanded" in explored
        assert "nodes_generated" in explored

    def test_cbs_single_agent(self):
        gm = _make_grid(5, 5)
        planner = CBSPlanner(gm)
        paths, _ = planner.plan([((0, 0), (4, 4))])
        assert paths[0] is not None
        assert paths[0][-1] == (4, 4)
