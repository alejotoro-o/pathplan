from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

from pathplan.core.map import GridMap

Path = List[Tuple[int, int]]
Agents = List[Tuple[Tuple[int, int], Tuple[int, int]]]


class BaseMAPFSolver(ABC):
    """Abstract base class for Multi-Agent Path Finding solvers.

    Subclasses implement :meth:`plan` which computes collision-free
    paths for a set of agents sharing a common grid.

    Parameters
    ----------
    grid_map:
        The shared occupancy grid.
    max_timesteps:
        Upper bound on the planning horizon (used by the low-level
        search and as a safety cut-off).
    """

    def __init__(self, grid_map: GridMap, max_timesteps: int = 500) -> None:
        self.grid_map = grid_map
        self.max_timesteps = max_timesteps

    @abstractmethod
    def plan(self, agents: Agents) -> Tuple[List[Optional[Path]], Any]:
        """Compute collision-free paths for all *agents*.

        Parameters
        ----------
        agents:
            List of ``(start, goal)`` tuples.  **List order defines
            priority**: agent 0 has the highest priority for
            priority-based algorithms.

        Returns
        -------
        (paths, explored):
            ``paths`` is a list of length ``len(agents)``.  Each
            element is either a list of ``(row, col)`` positions from
            start to goal, or ``None`` if that agent could not be
            planned.  ``explored`` is algorithm-specific diagnostic
            data.
        """
        pass
