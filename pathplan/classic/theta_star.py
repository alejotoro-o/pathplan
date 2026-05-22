import heapq
import numpy as np
from typing import Tuple, List, Optional, Set, Dict, Generator

from pathplan.core.base_solver import BaseSolver
from pathplan.core.map import GridMap
from pathplan.utils.geometry import edge_collision_free

class ThetaStarPlanner(BaseSolver):
    """Any-angle graph planning solver (Theta*)."""

    def __init__(self, grid_map: GridMap, occupancy_threshold: float = 0.5):
        super().__init__(grid_map)
        self.threshold = occupancy_threshold

    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return float(np.linalg.norm(np.array(p1) - np.array(p2)))

    def _get_neighbors(self, p: Tuple[int, int]) -> Generator[Tuple[int, int], None, None]:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                n = (p[0] + dx, p[1] + dy)
                if self.grid_map.is_valid_index(*n) and not self.grid_map.is_occupied(*n, threshold=self.threshold):
                    yield n

    def plan(
        self, start: Tuple[float, float], goal: Tuple[float, float]
    ) -> Tuple[Optional[List[Tuple[int, int]]], List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        
        # Standardize inputs to integer grid cells
        start_int = (int(start[0]), int(int(start[1])))
        goal_int = (int(goal[0]), int(int(goal[1])))

        g_score = {start_int: 0.0}
        parent = {start_int: start_int}
        
        # Priority Queue tracking: (f_score, cell_coordinates)
        open_set = [(self._distance(start_int, goal_int), start_int)]
        closed_set: Set[Tuple[int, int]] = set()
        explored_edges = []

        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            closed_set.add(current)

            if current == goal_int:
                path = [goal_int]
                while path[-1] != start_int:
                    path.append(parent[path[-1]])
                return path[::-1], explored_edges

            for n in self._get_neighbors(current):
                if n in closed_set:
                    continue
                
                curr_parent = parent[current]

                # Core modification: Evaluate if line-of-sight allows bypassing the intermediate node
                if curr_parent and edge_collision_free(curr_parent, n, self.grid_map, threshold=self.threshold):
                    new_g = g_score[curr_parent] + self._distance(curr_parent, n)
                    if new_g < g_score.get(n, float("inf")):
                        g_score[n] = new_g
                        parent[n] = curr_parent
                        f = new_g + self._distance(n, goal_int)
                        heapq.heappush(open_set, (f, n))
                        explored_edges.append((parent[n], n))
                else:
                    new_g = g_score[current] + self._distance(current, n)
                    if new_g < g_score.get(n, float("inf")):
                        g_score[n] = new_g
                        parent[n] = current
                        f = new_g + self._distance(n, goal_int)
                        heapq.heappush(open_set, (f, n))
                        explored_edges.append((parent[n], n))

        return None, explored_edges