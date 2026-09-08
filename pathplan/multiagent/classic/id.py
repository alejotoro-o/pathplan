from typing import Any, Dict, List, Optional, Tuple

from pathplan.core.map import GridMap
from pathplan.multiagent.core.base import Agents, BaseMAPFSolver, Path
from pathplan.multiagent.utils.utils import ReservationTable, detect_all_collisions, space_time_a_star


def _path_cost(path: List[Tuple[int, int]]) -> float:
    import math
    cost = 0.0
    for i in range(1, len(path)):
        dr = abs(path[i][0] - path[i - 1][0])
        dc = abs(path[i][1] - path[i - 1][1])
        cost += math.sqrt(2) if (dr + dc == 2) else 1.0
    return cost


class IDPlanner(BaseMAPFSolver):
    """Independence Detection — decomposes MAPF into independent sub-problems.

    Starts with singleton agent groups, solves each independently, and
    merges groups when their paths conflict.  The underlying solver
    can be any :class:`BaseMAPFSolver`.

    Parameters
    ----------
    grid_map:
        The shared occupancy grid.
    solver:
        The MAPF solver used for each independent group.  Must accept
        a list of ``(start, goal)`` pairs and return a list of paths.
    max_timesteps:
        Planning horizon (passed to the underlying solver).
    """

    def __init__(
        self,
        grid_map: GridMap,
        solver: Optional[BaseMAPFSolver] = None,
        max_timesteps: int = 500,
    ) -> None:
        super().__init__(grid_map, max_timesteps)
        if solver is None:
            from pathplan.multiagent.classic.prioritized import PrioritizedPlanner
            self._solver = PrioritizedPlanner(grid_map, max_timesteps)
        else:
            self._solver = solver

    def plan(self, agents: Agents) -> Tuple[List[Optional[Path]], Any]:
        n = len(agents)
        if n == 0:
            return [], {}

        # Initialise: each agent is its own group
        groups: List[List[int]] = [[i] for i in range(n)]
        # agent_id -> group_index
        agent_to_group: Dict[int, int] = {i: i for i in range(n)}

        # Cache of paths per agent
        paths: List[Optional[Path]] = [None] * n

        # Solve each singleton group
        for g_idx, group in enumerate(groups):
            group_agents = [agents[i] for i in group]
            group_paths, _ = self._solver.plan(group_agents)
            for k, agent_id in enumerate(group):
                paths[agent_id] = group_paths[k]

        # Merge conflicting groups until no collisions remain
        changed = True
        while changed:
            changed = False
            g = len(groups)
            for gi in range(g):
                for gj in range(gi + 1, g):
                    if not groups[gi] or not groups[gj]:
                        continue
                    # Check if any agent in gi conflicts with any in gj
                    gi_paths = [paths[i] for i in groups[gi] if paths[i] is not None]
                    gj_paths = [paths[i] for i in groups[gj] if paths[i] is not None]
                    if not gi_paths or not gj_paths:
                        continue
                    all_paths = gi_paths + gj_paths
                    collisions = detect_all_collisions(all_paths)
                    if collisions:
                        # Merge gj into gi
                        merged = groups[gi] + groups[gj]
                        groups[gi] = merged
                        for agent_id in merged:
                            agent_to_group[agent_id] = gi
                        groups[gj] = []
                        # Replan merged group
                        merged_agents = [agents[i] for i in merged]
                        merged_paths, _ = self._solver.plan(merged_agents)
                        for k, agent_id in enumerate(merged):
                            paths[agent_id] = merged_paths[k]
                        changed = True
                        break
                if changed:
                    break

        explored: Dict[str, Any] = {
            "num_groups": sum(1 for g in groups if g),
            "group_sizes": [len(g) for g in groups if g],
        }
        return paths, explored
