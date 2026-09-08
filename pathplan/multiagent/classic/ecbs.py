import heapq
from typing import Any, Dict, List, Optional, Set, Tuple

from pathplan.core.map import GridMap
from pathplan.multiagent.core.base import Agents, BaseMAPFSolver, Path
from pathplan.multiagent.utils.utils import (
    ReservationTable,
    build_constraint_table,
    detect_all_collisions,
    space_time_a_star,
)

# Reuse helpers from CBS
from pathplan.multiagent.classic.cbs import _CTNode, _path_cost, _standard_splitting


def _focal_node_cost(node: _CTNode) -> int:
    """Secondary ordering: number of collisions (used by focal search)."""
    return len(detect_all_collisions(node.paths))


class ECBSPlanner(BaseMAPFSolver):
    """Enhanced Conflict-Based Search — bounded-suboptimal for sum-of-costs.

    Like CBS but uses *focal search* at both levels to find solutions
    within ``w × optimal_cost`` much faster than optimal CBS.

    Parameters
    ----------
    grid_map:
        The shared occupancy grid.
    max_timesteps:
        Planning horizon for each low-level search.
    w:
        Suboptimality bound.  ``w = 1.0`` gives optimal CBS.
        ``w > 1.0`` allows cheaper solutions at the cost of quality.
    """

    def __init__(
        self,
        grid_map: GridMap,
        max_timesteps: int = 500,
        w: float = 1.5,
    ) -> None:
        super().__init__(grid_map, max_timesteps)
        self.w = max(w, 1.0)

    # ── low-level: bounded-suboptimal space-time A* ────────────────────

    def _replan_agent_bounded(
        self,
        agent_id: int,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        constraints: List[Dict],
    ) -> Optional[Path]:
        """Find a path within ``w × optimal`` cost subject to constraints."""
        rt = ReservationTable()
        ct = build_constraint_table(constraints, agent_id)

        inf_cons: Dict[Tuple[int, int], int] = {}
        for t, clist in ct.items():
            for c in clist:
                if c["type"] == "inf":
                    pos = c["loc"][0]
                    prev = inf_cons.get(pos)
                    if prev is None or t < prev:
                        inf_cons[pos] = t

        # Optimal path (for focal threshold)
        opt = space_time_a_star(
            self.grid_map, start, goal, rt,
            max_timesteps=self.max_timesteps,
            constraint_table=ct,
            inf_constraints=inf_cons if inf_cons else None,
        )
        if opt is None:
            return None
        if self.w <= 1.0:
            return opt

        # Bounded-suboptimal path via focal search on the low level
        return self._focal_low_level(
            start, goal, rt, ct, inf_cons, _path_cost(opt)
        )

    def _focal_low_level(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        rt: ReservationTable,
        ct: Dict[int, List[Dict]],
        inf_cons: Dict[Tuple[int, int], int],
        optimal_cost: float,
    ) -> Optional[Path]:
        """Low-level focal A* that returns a path within w × optimal_cost."""
        from pathplan.multiagent.utils.utils import (
            SQRT2,
            _ACTIONS,
            _ACTION_COSTS,
            _corner_cutting_free,
            _is_valid,
            _octile_heuristic,
        )

        sr, sc = start
        gr, gc = goal
        threshold = self.w * optimal_cost

        root = {
            "pos": start, "t": 0, "g": 0.0,
            "f": _octile_heuristic(sr, sc, gr, gc),
            "parent": None,
        }

        main_open: list = []
        heapq.heappush(main_open, (root["f"], root["g"], root["pos"], 0, root))
        focal_open: list = []  # entries: (num_collisions, f, g, pos, t, node)
        closed: Set[Tuple[int, int, int]] = set()
        best_g: Dict[Tuple[int, int, int], float] = {(*start, 0): 0.0}
        best_solution: Optional[dict] = None
        best_cost = float("inf")

        while main_open or focal_open:
            # Pick from focal if available, else main
            if focal_open:
                _, _, _, _, _, node = heapq.heappop(focal_open)
            else:
                _, _, _, _, node = heapq.heappop(main_open)

            state = (*node["pos"], node["t"])
            if state in closed:
                continue
            closed.add(state)

            if node["pos"] == goal:
                if node["g"] < best_cost:
                    best_cost = node["g"]
                    best_solution = node
                continue

            if node["t"] >= self.max_timesteps:
                continue

            r, c = node["pos"]
            nt = node["t"] + 1

            for idx, (dr, dc) in enumerate(_ACTIONS):
                nr, nc = r + dr, c + dc
                cost = _ACTION_COSTS[idx]

                if not _is_valid(self.grid_map, nr, nc):
                    continue
                if dr != 0 and dc != 0 and not _corner_cutting_free(self.grid_map, r, c, dr, dc):
                    continue
                if rt.is_reserved(nr, nc, nt):
                    continue
                if rt.is_edge_reserved(node["pos"], (nr, nc), nt):
                    continue
                from pathplan.multiagent.utils.utils import is_constrained, _is_inf_constrained
                if is_constrained(node["pos"], (nr, nc), nt, ct):
                    continue
                if _is_inf_constrained((nr, nc), nt, inf_cons):
                    continue

                ng = node["g"] + cost
                ns = (nr, nc, nc, nt)
                # ns is (row, col, timestep) — but we use (nr, nc, nt)
                ns_key = (nr, nc, nt)
                if ns_key in closed:
                    continue
                prev = best_g.get(ns_key)
                if prev is not None and ng >= prev:
                    continue
                best_g[ns_key] = ng

                child = {
                    "pos": (nr, nc), "t": nt, "g": ng,
                    "f": ng + _octile_heuristic(nr, nc, gr, gc),
                    "parent": node,
                }

                # Count collisions with all other agents' paths is not possible
                # here — use g-cost relative to threshold as focal criterion
                in_focal = child["f"] <= threshold
                if in_focal:
                    # Use negative g as tie-breaker (higher g = explored first)
                    heapq.heappush(focal_open, (-ng, child["f"], ng, (nr, nc), nt, child))
                else:
                    heapq.heappush(main_open, (child["f"], ng, (nr, nc), nt, child))

        if best_solution is not None:
            path: List[Tuple[int, int]] = []
            n: Optional[dict] = best_solution
            while n is not None:
                path.append(n["pos"])
                n = n["parent"]
            path.reverse()
            return path
        return None

    # ── high-level: focal search on CT ─────────────────────────────────

    def plan(self, agents: Agents) -> Tuple[List[Optional[Path]], Any]:
        n = len(agents)
        if n == 0:
            return [], {}

        # Root: independent planning
        rt = ReservationTable()
        root_paths: List[Optional[Path]] = []
        for start, goal in agents:
            p = space_time_a_star(
                self.grid_map, start, goal, rt,
                max_timesteps=self.max_timesteps,
            )
            root_paths.append(p)

        if any(p is None for p in root_paths):
            return root_paths, {"nodes_expanded": 0, "nodes_generated": 0}

        root_cost = sum(_path_cost(p) for p in root_paths)
        root = _CTNode([], root_paths, root_cost)

        collisions = detect_all_collisions(root.paths)
        if not collisions:
            return root_paths, {"nodes_expanded": 0, "nodes_generated": 0}

        # Compute lower bound = root cost (sum of individual optima)
        lower_bound = root_cost

        # Main open (ordered by cost) and focal open (ordered by collisions)
        main_open: list = []
        focal_open: list = []
        heapq.heappush(main_open, root)
        nodes_expanded = 0
        nodes_generated = 1

        while main_open or focal_open:
            if focal_open:
                _, node = heapq.heappop(focal_open)
            else:
                node = heapq.heappop(main_open)

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

                new_path = self._replan_agent_bounded(
                    agent_id, start, goal, child_constraints
                )
                if new_path is None:
                    continue

                child_paths = list(node.paths)
                child_paths[agent_id] = new_path
                child_cost = sum(_path_cost(p) for p in child_paths)
                child = _CTNode(child_constraints, child_paths, child_cost)

                num_col = len(detect_all_collisions(child.paths))
                if child_cost <= self.w * lower_bound:
                    heapq.heappush(focal_open, (num_col, child))
                else:
                    heapq.heappush(main_open, child)
                nodes_generated += 1

        return [None] * n, {
            "nodes_expanded": nodes_expanded,
            "nodes_generated": nodes_generated,
        }
