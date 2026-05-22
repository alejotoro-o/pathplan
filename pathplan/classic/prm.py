import random
import numpy as np
import heapq
from typing import Tuple, List, Optional, Dict, Set

from pathplan.core.base_solver import BaseSolver
from pathplan.core.map import GridMap
from pathplan.utils.geometry import edge_collision_free

class PRMPlanner(BaseSolver):
    """Probabilistic Roadmap (PRM) path planner."""

    def __init__(
        self,
        grid_map: GridMap,
        num_samples: int = 200,
        connection_radius: float = 5.0,
        occupancy_threshold: float = 0.5
    ):
        super().__init__(grid_map)
        self.num_samples = num_samples
        self.connection_radius = connection_radius
        self.threshold = occupancy_threshold

    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return float(np.linalg.norm(np.array(p1) - np.array(p2)))

    def plan(
        self, start: Tuple[float, float], goal: Tuple[float, float]
    ) -> Tuple[Optional[List[Tuple[float, float]]], List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
        
        # 1. Sampling
        samples = [start, goal]
        for _ in range(self.num_samples):
            sample = (
                random.uniform(0, self.grid_map.height - 1),
                random.uniform(0, self.grid_map.width - 1)
            )
            if not self.grid_map.is_occupied(int(sample[0]), int(sample[1]), self.threshold):
                samples.append(sample)

        # 2. Roadmap construction
        roadmap: Dict[Tuple[float, float], List[Tuple[float, float]]] = {s: [] for s in samples}
        explored_edges = []
        for i, s1 in enumerate(samples):
            for s2 in samples[i+1:]:
                dist = self._distance(s1, s2)
                if dist < self.connection_radius:
                    if edge_collision_free(s1, s2, self.grid_map, self.threshold):
                        roadmap[s1].append(s2)
                        roadmap[s2].append(s1)
                        explored_edges.append((s1, s2))

        # 3. Search on roadmap (A*)
        path = self._astar_search(roadmap, start, goal)
        return path, explored_edges

    def _astar_search(
        self, roadmap: Dict[Tuple[float, float], List[Tuple[float, float]]], 
        start: Tuple[float, float], goal: Tuple[float, float]
    ) -> Optional[List[Tuple[float, float]]]:
        open_set = [(self._distance(start, goal), start)]
        g_score = {start: 0.0}
        parent = {start: None}
        closed_set = set()

        while open_set:
            _, current = heapq.heappop(open_set)
            if current in closed_set: continue
            closed_set.add(current)

            if current == goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = parent[current]
                return path[::-1]

            for neighbor in roadmap[current]:
                if neighbor in closed_set: continue
                new_g = g_score[current] + self._distance(current, neighbor)
                if neighbor not in g_score or new_g < g_score[neighbor]:
                    g_score[neighbor] = new_g
                    parent[neighbor] = current
                    f = new_g + self._distance(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))
        
        return None
