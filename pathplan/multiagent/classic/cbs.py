import heapq
from typing import Any, Dict, List, Optional, Tuple

from pathplan.core.map import GridMap
from pathplan.multiagent.core.base import Agents, BaseMAPFSolver, Path
from pathplan.multiagent.utils.utils import (
    ReservationTable,
    build_constraint_table,
    detect_all_collisions,
    space_time_a_star,
)


def _path_cost(path: List[Tuple[int, int]]) -> float:
    """Sum of step costs along *path*."""
    import math

    cost = 0.0
    for i in range(1, len(path)):
        dr = abs(path[i][0] - path[i - 1][0])
        dc = abs(path[i][1] - path[i - 1][1])
        cost += math.sqrt(2) if (dr + dc == 2) else 1.0
    return cost


def _standard_splitting(collision: Dict) -> List[Dict]:
    """Generate two constraints that resolve *collision*."""
    a1, a2 = collision["a1"], collision["a2"]
    loc = collision["loc"]
    t = collision["timestep"]
    ctype = collision["type"]

    if ctype == "vertex":
        return [
            {"agent": a1, "loc": [loc[0]], "timestep": t, "type": "vertex"},
            {"agent": a2, "loc": [loc[0]], "timestep": t, "type": "vertex"},
        ]
    # edge
    return [
        {"agent": a1, "loc": loc, "timestep": t, "type": "edge"},
        {"agent": a2, "loc": [loc[1], loc[0]], "timestep": t, "type": "edge"},
    ]


class _CTNode:
    """A node in the CBS constraint tree."""

    __slots__ = ("constraints", "paths", "cost")

    def __init__(
        self,
        constraints: List[Dict],
        paths: List[Optional[Path]],
        cost: float,
    ) -> None:
        self.constraints = constraints
        self.paths = paths
        self.cost = cost

    def __lt__(self, other: "_CTNode") -> bool:
        return self.cost < other.cost


class CBSPlanner(BaseMAPFSolver):
    """Conflict-Based Search — optimal for sum-of-costs.

    Two-level algorithm: the high level searches a binary constraint
    tree (CT) and the low level replans individual agents with
    space-time A* subject to accumulated constraints.

    Parameters
    ----------
    grid_map:
        The shared occupancy grid.
    max_timesteps:
        Planning horizon for each low-level search.
    """

    def __init__(self, grid_map: GridMap, max_timesteps: int = 500) -> None:
        super().__init__(grid_map, max_timesteps)

    def _replan_agent(
        self,
        agent_id: int,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        constraints: List[Dict],
    ) -> Optional[Path]:
        """Replan a single agent subject to *constraints*."""
        rt = ReservationTable()
        ct = build_constraint_table(constraints, agent_id)

        # Extract "inf" constraints from the constraint table
        inf_cons: Dict[Tuple[int, int], int] = {}
        for t, clist in ct.items():
            for c in clist:
                if c["type"] == "inf":
                    pos = c["loc"][0]
                    prev = inf_cons.get(pos)
                    if prev is None or t < prev:
                        inf_cons[pos] = t

        return space_time_a_star(
            self.grid_map, start, goal, rt,
            max_timesteps=self.max_timesteps,
            constraint_table=ct,
            inf_constraints=inf_cons if inf_cons else None,
        )

    def plan(self, agents: Agents) -> Tuple[List[Optional[Path]], Any]:
        n = len(agents)
        if n == 0:
            return [], {}

        # ── root node: plan each agent independently ───────────────────
        rt = ReservationTable()
        root_paths: List[Optional[Path]] = []
        for start, goal in agents:
            p = space_time_a_star(
                self.grid_map, start, goal, rt,
                max_timesteps=self.max_timesteps,
            )
            root_paths.append(p)

        # If any agent is unreachable, return immediately
        if any(p is None for p in root_paths):
            return root_paths, {"nodes_expanded": 0, "nodes_generated": 0}

        collisions = detect_all_collisions(root_paths)
        root_cost = sum(_path_cost(p) for p in root_paths)
        root = _CTNode([], root_paths, root_cost)

        if not collisions:
            return root_paths, {"nodes_expanded": 0, "nodes_generated": 0}

        # ── high-level search ──────────────────────────────────────────
        open_list: list = []
        heapq.heappush(open_list, root)
        nodes_expanded = 0
        nodes_generated = 1

        while open_list:
            node = heapq.heappop(open_list)
            nodes_expanded += 1

            collisions = detect_all_collisions(node.paths)
            if not collisions:
                return node.paths, {
                    "nodes_expanded": nodes_expanded,
                    "nodes_generated": nodes_generated,
                }

            first = collisions[0]
            constraints = _standard_splitting(first)

            for constraint in constraints:
                child_constraints = node.constraints + [constraint]
                agent_id = constraint["agent"]
                start, goal = agents[agent_id]

                new_path = self._replan_agent(agent_id, start, goal, child_constraints)
                if new_path is None:
                    continue

                child_paths = list(node.paths)
                child_paths[agent_id] = new_path
                child_cost = sum(_path_cost(p) for p in child_paths)

                child = _CTNode(child_constraints, child_paths, child_cost)
                heapq.heappush(open_list, child)
                nodes_generated += 1

        return [None] * n, {
            "nodes_expanded": nodes_expanded,
            "nodes_generated": nodes_generated,
        }
