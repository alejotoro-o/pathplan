import math
import numpy as np
import pytest
from pathplan.core.map import GridMap
from pathplan.multiagent import (
    CBSPlanner,
    ECBSPlanner,
    IDPlanner,
    IndependentPlanner,
    PBSPlanner,
    PrioritizedPlanner,
)
from pathplan.multiagent.utils import detect_collision


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


# ── ECBS ──────────────────────────────────────────────────────────────────


class TestECBS:
    def test_single_agent(self):
        gm = _make_grid(5, 5)
        planner = ECBSPlanner(gm)
        paths, _ = planner.plan([((0, 0), (4, 4))])
        assert paths[0] is not None
        assert paths[0][-1] == (4, 4)

    def test_two_agents_collision_free(self):
        gm = _make_grid(5, 5)
        planner = ECBSPlanner(gm)
        agents = [((0, 0), (4, 4)), ((4, 0), (0, 4))]
        paths, _ = planner.plan(agents)
        assert all(p is not None for p in paths)
        assert detect_collision(paths[0], paths[1]) is None

    def test_bounded_suboptimal(self):
        """ECBS with w > 1 should find a solution at least as cheap as CBS."""
        gm = _make_grid(5, 5)
        agents = [((0, 0), (4, 4)), ((4, 0), (0, 4)), ((0, 4), (4, 0))]

        cbs = CBSPlanner(gm)
        cbs_paths, _ = cbs.plan(agents)
        cbs_cost = sum(_path_length(p) for p in cbs_paths)

        ecbs = ECBSPlanner(gm, w=2.0)
        ecbs_paths, _ = ecbs.plan(agents)
        assert all(p is not None for p in ecbs_paths)
        ecbs_cost = sum(_path_length(p) for p in ecbs_paths)

        # ECBS cost should be <= w × CBS cost
        assert ecbs_cost <= 2.0 * cbs_cost + 1e-9

    def test_w_equals_one_is_cbs(self):
        """ECBS with w=1.0 should produce the same cost as CBS."""
        gm = _make_grid(3, 3)
        agents = [((0, 0), (2, 2)), ((2, 0), (0, 2))]

        cbs = CBSPlanner(gm)
        cbs_paths, _ = cbs.plan(agents)
        cbs_cost = sum(_path_length(p) for p in cbs_paths)

        ecbs = ECBSPlanner(gm, w=1.0)
        ecbs_paths, _ = ecbs.plan(agents)
        ecbs_cost = sum(_path_length(p) for p in ecbs_paths)

        assert abs(cbs_cost - ecbs_cost) < 1e-9

    def test_explored_data(self):
        gm = _make_grid(3, 3)
        planner = ECBSPlanner(gm)
        agents = [((0, 0), (2, 2)), ((2, 0), (0, 2))]
        _, explored = planner.plan(agents)
        assert "nodes_expanded" in explored
        assert "nodes_generated" in explored


# ── Independence Detection ────────────────────────────────────────────────


class TestIndependenceDetection:
    def test_single_agent(self):
        gm = _make_grid(5, 5)
        planner = IDPlanner(gm)
        paths, _ = planner.plan([((0, 0), (4, 4))])
        assert paths[0] is not None
        assert paths[0][-1] == (4, 4)

    def test_independent_groups_separated(self):
        """Two agents in separate areas should stay independent."""
        gm = _make_grid(5, 10)
        planner = IDPlanner(gm)
        agents = [
            ((0, 0), (0, 4)),   # top-left
            ((4, 5), (4, 9)),   # bottom-right
        ]
        paths, explored = planner.plan(agents)
        assert all(p is not None for p in paths)
        # Should have 2 groups (no merging needed)
        assert explored["num_groups"] == 2

    def test_conflicting_agents_merged(self):
        """Two agents crossing should be merged into one group."""
        gm = _make_grid(3, 3)
        planner = IDPlanner(gm)
        agents = [
            ((0, 0), (2, 2)),
            ((2, 0), (0, 2)),
        ]
        paths, explored = planner.plan(agents)
        assert all(p is not None for p in paths)
        # Should be 1 group (merged due to conflict)
        assert explored["num_groups"] == 1

    def test_collision_free_result(self):
        gm = _make_grid(5, 5)
        planner = IDPlanner(gm)
        agents = [((0, 0), (4, 4)), ((4, 0), (0, 4))]
        paths, _ = planner.plan(agents)
        if all(p is not None for p in paths):
            assert detect_collision(paths[0], paths[1]) is None

    def test_with_cbs_solver(self):
        """ID can use CBS as the underlying solver."""
        gm = _make_grid(3, 3)
        cbs = CBSPlanner(gm)
        planner = IDPlanner(gm, solver=cbs)
        agents = [((0, 0), (2, 2)), ((2, 0), (0, 2))]
        paths, explored = planner.plan(agents)
        assert all(p is not None for p in paths)
        assert detect_collision(paths[0], paths[1]) is None


# ── PBS ───────────────────────────────────────────────────────────────────


class TestPBS:
    def test_single_agent(self):
        gm = _make_grid(5, 5)
        planner = PBSPlanner(gm)
        paths, _ = planner.plan([((0, 0), (4, 4))])
        assert paths[0] is not None
        assert paths[0][-1] == (4, 4)

    def test_two_agents_collision_free(self):
        gm = _make_grid(3, 3)
        planner = PBSPlanner(gm)
        agents = [((0, 0), (2, 2)), ((2, 0), (0, 2))]
        paths, explored = planner.plan(agents)
        assert all(p is not None for p in paths)
        assert detect_collision(paths[0], paths[1]) is None

    def test_more_complete_than_prioritized(self):
        """PBS should handle cases where PrioritizedPlanner fails."""
        gm = _make_grid(3, 3)
        # This scenario: one ordering works, the other doesn't
        agents = [((0, 0), (2, 2)), ((2, 0), (0, 2))]

        pri = PrioritizedPlanner(gm)
        pri_paths, _ = pri.plan(agents)

        pbs = PBSPlanner(gm)
        pbs_paths, _ = pbs.plan(agents)
        assert all(p is not None for p in pbs_paths)

    def test_explored_data(self):
        gm = _make_grid(3, 3)
        planner = PBSPlanner(gm)
        agents = [((0, 0), (2, 2)), ((2, 0), (0, 2))]
        _, explored = planner.plan(agents)
        assert "nodes_expanded" in explored
        assert "nodes_generated" in explored


# ── Shared: all six planners ──────────────────────────────────────────────


ALL_PLANNERS = [
    IndependentPlanner,
    PrioritizedPlanner,
    CBSPlanner,
    ECBSPlanner,
    IDPlanner,
    PBSPlanner,
]


@pytest.mark.parametrize("planner_cls", ALL_PLANNERS)
class TestAllPlanners:
    def test_single_agent(self, planner_cls):
        gm = _make_grid(5, 5)
        planner = planner_cls(gm)
        paths, _ = planner.plan([((0, 0), (4, 4))])
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
        agents = [((0, 0), (0, 4)), ((4, 0), (4, 4))]
        paths, _ = planner.plan(agents)
        assert all(p is not None for p in paths)

    def test_empty_agents(self, planner_cls):
        gm = _make_grid(5, 5)
        planner = planner_cls(gm)
        paths, _ = planner.plan([])
        assert paths == []
