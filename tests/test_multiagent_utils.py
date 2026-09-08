import math
import numpy as np
import pytest
from pathplan.core.map import GridMap
from pathplan.multiagent import (
    ReservationTable,
    build_constraint_table,
    detect_all_collisions,
    detect_collision,
    is_constrained,
    space_time_a_star,
)


# ── ReservationTable ──────────────────────────────────────────────────────


class TestReservationTable:
    def test_reserve_and_check(self):
        rt = ReservationTable()
        assert not rt.is_reserved(0, 0, 0)
        rt.reserve_vertex(0, 0, 0)
        assert rt.is_reserved(0, 0, 0)
        assert not rt.is_reserved(0, 1, 0)

    def test_reserve_path(self):
        rt = ReservationTable()
        path = [(0, 0), (0, 1), (1, 1)]
        rt.reserve_path(path)
        for t, (r, c) in enumerate(path):
            assert rt.is_reserved(r, c, t)

    def test_goal_parking(self):
        rt = ReservationTable()
        path = [(0, 0), (0, 1), (0, 2)]
        rt.reserve_path(path)
        for t in range(2, 10):
            assert rt.is_reserved(0, 2, t), f"goal not parked at t={t}"

    def test_edge_reserved(self):
        rt = ReservationTable()
        rt.reserve_edge((0, 0), (0, 1), 1)
        assert rt.is_edge_reserved((0, 0), (0, 1), 1)
        assert rt.is_edge_reserved((0, 1), (0, 0), 1)
        assert not rt.is_edge_reserved((0, 0), (0, 1), 2)

    def test_path_edges_reserved(self):
        rt = ReservationTable()
        path = [(0, 0), (0, 1), (1, 1)]
        rt.reserve_path(path)
        assert rt.is_edge_reserved((0, 0), (0, 1), 1)
        assert rt.is_edge_reserved((0, 1), (1, 1), 2)


# ── Collision detection ───────────────────────────────────────────────────


class TestDetectCollision:
    def test_vertex_collision(self):
        p1 = [(0, 0), (0, 1), (0, 2)]
        p2 = [(0, 2), (0, 1), (0, 0)]
        c = detect_collision(p1, p2)
        assert c is not None
        assert c["type"] == "vertex"
        assert c["loc"] == [(0, 1)]

    def test_edge_collision(self):
        p1 = [(0, 0), (0, 1), (0, 2)]
        p2 = [(0, 2), (0, 1), (0, 0)]
        c = detect_collision(p1, p2)
        assert c is not None

    def test_no_collision(self):
        p1 = [(0, 0), (0, 1), (0, 2)]
        p2 = [(2, 0), (2, 1), (2, 2)]
        assert detect_collision(p1, p2) is None

    def test_detect_all_collisions(self):
        p1 = [(0, 0), (0, 1)]
        p2 = [(0, 1), (0, 0)]
        p3 = [(2, 2), (2, 3)]
        cols = detect_all_collisions([p1, p2, p3])
        assert len(cols) == 1
        assert cols[0]["a1"] == 0
        assert cols[0]["a2"] == 1


# ── Constraint helpers ────────────────────────────────────────────────────


class TestConstraintHelpers:
    def test_build_constraint_table(self):
        constraints = [
            {"agent": 0, "loc": [(1, 1)], "timestep": 3, "type": "vertex"},
            {"agent": 1, "loc": [(0, 0)], "timestep": 2, "type": "vertex"},
            {"agent": 0, "loc": [(2, 2), (2, 3)], "timestep": 5, "type": "edge"},
        ]
        table = build_constraint_table(constraints, agent_id=0)
        assert 3 in table
        assert len(table[3]) == 1
        assert table[3][0]["type"] == "vertex"
        assert 5 in table
        assert table[5][0]["type"] == "edge"
        assert 2 not in table  # belongs to agent 1

    def test_is_constrained_vertex(self):
        table = {2: [{"type": "vertex", "loc": [(1, 1)], "timestep": 2}]}
        assert is_constrained((0, 0), (1, 1), 2, table)
        assert not is_constrained((0, 0), (1, 0), 2, table)

    def test_is_constrained_edge(self):
        table = {3: [{"type": "edge", "loc": [(0, 0), (0, 1)], "timestep": 3}]}
        assert is_constrained((0, 0), (0, 1), 3, table)
        assert not is_constrained((0, 0), (1, 0), 3, table)

    def test_no_constraints(self):
        assert not is_constrained((0, 0), (1, 1), 5, {})


# ── Space-Time A* ─────────────────────────────────────────────────────────


class TestSpaceTimeAStar:
    def _make_grid(self, h, w, obstacles=None):
        data = np.zeros((h, w), dtype=np.float32)
        if obstacles:
            for r, c in obstacles:
                data[r, c] = 1.0
        return GridMap(data)

    def test_simple_path_empty(self):
        gm = self._make_grid(5, 5)
        rt = ReservationTable()
        path = space_time_a_star(gm, (0, 0), (4, 4), rt)
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (4, 4)

    def test_avoids_reservation(self):
        gm = self._make_grid(3, 5)
        rt = ReservationTable()
        rt.reserve_vertex(0, 2, 0)
        rt.reserve_vertex(0, 2, 1)
        path = space_time_a_star(gm, (0, 0), (0, 4), rt, max_timesteps=20)
        assert path is not None
        for t, (r, c) in enumerate(path):
            assert not rt.is_reserved(r, c, t)

    def test_blocked_returns_none(self):
        gm = self._make_grid(3, 3, obstacles=[(0, 1), (1, 1), (2, 1)])
        rt = ReservationTable()
        path = space_time_a_star(gm, (0, 0), (0, 2), rt)
        assert path is None

    def test_wait_action(self):
        gm = self._make_grid(1, 3)
        rt = ReservationTable()
        rt.reserve_vertex(0, 2, 0)
        rt.reserve_vertex(0, 2, 1)
        rt.reserve_vertex(0, 2, 2)
        path = space_time_a_star(gm, (0, 0), (0, 2), rt, max_timesteps=10)
        assert path is not None
        assert path[-1] == (0, 2)

    def test_start_equals_goal(self):
        gm = self._make_grid(3, 3)
        rt = ReservationTable()
        path = space_time_a_star(gm, (1, 1), (1, 1), rt)
        assert path == [(1, 1)]

    def test_diagonal_path(self):
        gm = self._make_grid(5, 5)
        rt = ReservationTable()
        path = space_time_a_star(gm, (0, 0), (2, 2), rt)
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (2, 2)
        diag_cost = math.sqrt(2)
        assert abs(path_length(path) - 2 * diag_cost) < 1e-9

    def test_blocked_start(self):
        gm = self._make_grid(3, 3, obstacles=[(0, 0)])
        rt = ReservationTable()
        path = space_time_a_star(gm, (0, 0), (2, 2), rt)
        assert path is None

    def test_blocked_goal(self):
        gm = self._make_grid(3, 3, obstacles=[(2, 2)])
        rt = ReservationTable()
        path = space_time_a_star(gm, (0, 0), (2, 2), rt)
        assert path is None

    def test_avoids_vertex_constraint(self):
        gm = self._make_grid(3, 3)
        rt = ReservationTable()
        ct = build_constraint_table(
            [{"agent": 0, "loc": [(1, 1)], "timestep": 1, "type": "vertex"}], 0
        )
        path = space_time_a_star(gm, (0, 0), (2, 2), rt, constraint_table=ct)
        assert path is not None
        # Path must avoid (1,1) at t=1
        if len(path) > 1:
            assert path[1] != (1, 1)

    def test_avoids_edge_constraint(self):
        gm = self._make_grid(1, 3)
        rt = ReservationTable()
        ct = build_constraint_table(
            [{"agent": 0, "loc": [(0, 0), (0, 1)], "timestep": 1, "type": "edge"}], 0
        )
        path = space_time_a_star(gm, (0, 0), (0, 2), rt, constraint_table=ct)
        assert path is not None
        # First move must not be (0,0)->(0,1)
        assert path[1] != (0, 1)


def path_length(path):
    total = 0.0
    for i in range(1, len(path)):
        dr = abs(path[i][0] - path[i - 1][0])
        dc = abs(path[i][1] - path[i - 1][1])
        if dr + dc == 2:
            total += math.sqrt(2)
        else:
            total += 1.0
    return total
