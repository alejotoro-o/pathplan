import heapq
import math
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Set, Tuple

from pathplan.core.map import GridMap

# 8-directional movement + WAIT.  Cost: cardinal=1, diagonal=√2, wait=1.
_DIRECTIONS: List[Tuple[int, int]] = [
    (-1, 0),   # N
    (-1, 1),   # NE
    (0, 1),    # E
    (1, 1),    # SE
    (1, 0),    # S
    (1, -1),   # SW
    (0, -1),   # W
    (-1, -1),  # NW
]
_WAIT = (0, 0)
_ACTIONS = _DIRECTIONS + [_WAIT]
_CARDINAL_COST = 1.0
_DIAG_COST = math.sqrt(2.0)
_WAIT_COST = 1.0
_ACTION_COSTS = [_CARDINAL_COST] * 4 + [_DIAG_COST] * 4 + [_WAIT_COST]

SQRT2 = math.sqrt(2.0)


def _octile_heuristic(r1: int, c1: int, r2: int, c2: int) -> float:
    dr = abs(r1 - r2)
    dc = abs(c1 - c2)
    return max(dr, dc) + (SQRT2 - 1.0) * min(dr, dc)


def _is_valid(grid_map: GridMap, row: int, col: int) -> bool:
    return grid_map.is_valid_index(row, col) and not grid_map.is_occupied(row, col)


def _corner_cutting_free(grid_map: GridMap, r: int, c: int, dr: int, dc: int) -> bool:
    """For a diagonal move (dr,dc), both adjacent cardinal cells must be free."""
    if dr != 0 and dc != 0:
        return _is_valid(grid_map, r + dr, c) and _is_valid(grid_map, r, c + dc)
    return True


def _reconstruct(node: dict) -> List[Tuple[int, int]]:
    path: List[Tuple[int, int]] = []
    n: Optional[dict] = node
    while n is not None:
        path.append(n["pos"])
        n = n["parent"]
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# ReservationTable
# ---------------------------------------------------------------------------

class ReservationTable:
    """Tracks space-time occupancy across agents.

    Vertex reservations: ``(row, col)`` occupied at a specific ``timestep``.
    Edge reservations:   move ``(from_pos, to_pos)`` forbidden at ``timestep``.
    Goal parking:       once an agent reaches its goal at time *T*, the goal
    cell is considered reserved for **all** ``t >= T``.
    """

    def __init__(self) -> None:
        self._vertex: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)
        self._edge: Dict[int, Set[Tuple[Tuple[int, int], Tuple[int, int]]]] = (
            defaultdict(set)
        )
        self._goal_arrival: Dict[Tuple[int, int], int] = {}

    # ---- queries ----------------------------------------------------------

    def is_reserved(self, row: int, col: int, timestep: int) -> bool:
        pos = (row, col)
        arrival = self._goal_arrival.get(pos)
        if arrival is not None and timestep >= arrival:
            return True
        return pos in self._vertex[timestep]

    def is_edge_reserved(
        self,
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int],
        timestep: int,
    ) -> bool:
        return (
            (from_pos, to_pos) in self._edge[timestep]
            or (to_pos, from_pos) in self._edge[timestep]
        )

    # ---- mutations --------------------------------------------------------

    def reserve_vertex(self, row: int, col: int, timestep: int) -> None:
        self._vertex[timestep].add((row, col))

    def reserve_edge(
        self,
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int],
        timestep: int,
    ) -> None:
        self._edge[timestep].add((from_pos, to_pos))

    def reserve_path(self, path: List[Tuple[int, int]]) -> None:
        """Reserve an entire path including goal parking."""
        for t, pos in enumerate(path):
            self.reserve_vertex(pos[0], pos[1], t)
            if t > 0:
                self.reserve_edge(path[t - 1], pos, t)
        if path:
            goal = path[-1]
            arrival = len(path) - 1
            prev = self._goal_arrival.get(goal)
            if prev is None or arrival < prev:
                self._goal_arrival[goal] = arrival


# ---------------------------------------------------------------------------
# Collision detection
# ---------------------------------------------------------------------------

def _get_location(path: List[Tuple[int, int]], t: int) -> Tuple[int, int]:
    if t < len(path):
        return path[t]
    return path[-1]


def detect_collision(
    path1: List[Tuple[int, int]],
    path2: List[Tuple[int, int]],
) -> Optional[Dict]:
    """Return the first collision between two paths, or ``None``."""
    max_t = max(len(path1), len(path2))
    for t in range(max_t):
        loc1 = _get_location(path1, t)
        loc2 = _get_location(path2, t)
        if loc1 == loc2:
            return {"loc": [loc1], "timestep": t, "type": "vertex"}
        if t > 0:
            prev1 = _get_location(path1, t - 1)
            prev2 = _get_location(path2, t - 1)
            if prev1 == loc2 and prev2 == loc1:
                return {"loc": [prev1, loc1], "timestep": t, "type": "edge"}
    return None


def detect_all_collisions(
    paths: List[List[Tuple[int, int]]],
) -> List[Dict]:
    """Return every collision across all agent pairs."""
    collisions: List[Dict] = []
    n = len(paths)
    for i in range(n):
        for j in range(i + 1, n):
            col = detect_collision(paths[i], paths[j])
            if col is not None:
                collisions.append({"a1": i, "a2": j, **col})
    return collisions


# ---------------------------------------------------------------------------
# Constraint table helpers (used by CBS)
# ---------------------------------------------------------------------------

def build_constraint_table(
    constraints: List[Dict], agent_id: int
) -> Dict[int, List[Dict]]:
    """Convert a flat constraint list into a per-timestep lookup for *agent_id*.

    Returns ``{timestep: [constraint, ...]}`` where each constraint dict
    has ``"type"`` (``"vertex"``, ``"edge"``, or ``"inf"``) and ``"loc"``.
    """
    table: Dict[int, List[Dict]] = {}
    for c in constraints:
        if c["agent"] != agent_id:
            continue
        t = c["timestep"]
        if t not in table:
            table[t] = []
        table[t].append(c)
    return table


def is_constrained(
    curr_pos: Tuple[int, int],
    next_pos: Tuple[int, int],
    next_time: int,
    constraint_table: Dict[int, List[Dict]],
) -> bool:
    """Check if moving from *curr_pos* to *next_pos* at *next_time* is forbidden."""
    if next_time in constraint_table:
        for c in constraint_table[next_time]:
            ctype = c["type"]
            if ctype == "vertex" and next_pos == c["loc"][0]:
                return True
            if ctype == "edge" and curr_pos == c["loc"][0] and next_pos == c["loc"][1]:
                return True
    return False


def _is_inf_constrained(
    next_pos: Tuple[int, int],
    next_time: int,
    inf_constraints: Dict[Tuple[int, int], int],
) -> bool:
    """Check ``"inf"`` (goal parking) constraints."""
    arrival = inf_constraints.get(next_pos)
    if arrival is not None and next_time >= arrival:
        return True
    return False


# ---------------------------------------------------------------------------
# Space-Time A*
# ---------------------------------------------------------------------------

def space_time_a_star(
    grid_map: GridMap,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    reservation_table: ReservationTable,
    max_timesteps: int = 500,
    constraint_table: Optional[Dict[int, List[Dict]]] = None,
    inf_constraints: Optional[Dict[Tuple[int, int], int]] = None,
) -> Optional[List[Tuple[int, int]]]:
    """A* in (row, col, time) space with 8-directional + wait actions.

    Returns a list of ``(row, col)`` positions or ``None`` if no
    collision-free path exists within *max_timesteps*.

    Parameters
    ----------
    constraint_table:
        Per-timestep constraint lookup (from :func:`build_constraint_table`).
    inf_constraints:
        Goal-parking constraints ``{pos: min_arrival}`` — position is
        forbidden at all ``t >= min_arrival``.
    """
    sr, sc = start
    gr, gc = goal

    if not _is_valid(grid_map, sr, sc) or not _is_valid(grid_map, gr, gc):
        return None
    if start == goal:
        return [start]

    root = {
        "pos": start,
        "t": 0,
        "g": 0.0,
        "f": _octile_heuristic(sr, sc, gr, gc),
        "parent": None,
    }

    open_list: list = []
    heapq.heappush(open_list, (root["f"], root["g"], root["pos"], 0, root))
    closed: Set[Tuple[int, int, int]] = set()
    best_g: Dict[Tuple[int, int, int], float] = {(*start, 0): 0.0}

    while open_list:
        f, g, pos, t, node = heapq.heappop(open_list)
        state = (*pos, t)

        if state in closed:
            continue
        closed.add(state)

        if pos == goal:
            return _reconstruct(node)

        if t >= max_timesteps:
            continue

        r, c = pos
        nt = t + 1

        for idx, (dr, dc) in enumerate(_ACTIONS):
            nr, nc = r + dr, c + dc
            cost = _ACTION_COSTS[idx]

            if not _is_valid(grid_map, nr, nc):
                continue

            if dr != 0 and dc != 0 and not _corner_cutting_free(grid_map, r, c, dr, dc):
                continue

            if reservation_table.is_reserved(nr, nc, nt):
                continue

            if reservation_table.is_edge_reserved(pos, (nr, nc), nt):
                continue

            if constraint_table is not None and is_constrained(pos, (nr, nc), nt, constraint_table):
                continue

            if inf_constraints is not None and _is_inf_constrained((nr, nc), nt, inf_constraints):
                continue

            ng = g + cost
            ns = (nr, nc, nt)
            if ns in closed:
                continue
            prev = best_g.get(ns)
            if prev is not None and ng >= prev:
                continue
            best_g[ns] = ng

            child = {
                "pos": (nr, nc),
                "t": nt,
                "g": ng,
                "f": ng + _octile_heuristic(nr, nc, gr, gc),
                "parent": node,
            }
            heapq.heappush(open_list, (child["f"], ng, (nr, nc), nt, child))

    return None
