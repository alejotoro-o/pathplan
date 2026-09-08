from typing import Any, Dict, List, Optional, Tuple

from pathplan.core.map import GridMap
from pathplan.multiagent.core.base import Agents, BaseMAPFSolver, Path
from pathplan.multiagent.utils.utils import ReservationTable, space_time_a_star


class PrioritizedPlanner(BaseMAPFSolver):
    """Agents plan sequentially in list order.

    Each agent's path is reserved in the table, constraining all
    subsequent agents.  Priority is determined by list position:
    agent 0 has highest priority.

    Parameters
    ----------
    grid_map:
        The shared occupancy grid.
    max_timesteps:
        Planning horizon for each agent.
    """

    def __init__(self, grid_map: GridMap, max_timesteps: int = 500) -> None:
        super().__init__(grid_map, max_timesteps)

    def plan(self, agents: Agents) -> Tuple[List[Optional[Path]], Any]:
        rt = ReservationTable()
        paths: List[Optional[Path]] = []

        for start, goal in agents:
            path = space_time_a_star(
                self.grid_map, start, goal, rt,
                max_timesteps=self.max_timesteps,
            )
            paths.append(path)
            if path is not None:
                rt.reserve_path(path)

        explored: Dict[str, Any] = {
            "num_agents": len(agents),
            "all_found": all(p is not None for p in paths),
        }
        return paths, explored
