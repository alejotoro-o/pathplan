import heapq
import numpy as np
from typing import Tuple, List, Optional, Set, Dict

from pathplan.core.map import GridMap
from pathplan.core.base_solver import BaseSolver

class DijkstraPlanner(BaseSolver):
    """Classic 8-directional Dijkstra planning solver."""
    
    def __init__(self, grid_map: GridMap, occupancy_threshold: float = 0.5):
        super().__init__(grid_map)
        self.threshold = occupancy_threshold
        # 8-Directional Movements
        self.motion_vectors = [
            (0, -1), (0, 1), (-1, 0), (1, 0),   # Orthogonal
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # Diagonal
        ]

    def plan(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[Optional[List[Tuple[int, int]]], List[Tuple[int, int]]]:
        """
        Computes the shortest path using Dijkstra's algorithm.
        """
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))

        # Priority Queue storage tracker elements: (cost, position)
        open_heap: List[Tuple[float, Tuple[int, int]]] = [(0.0, start)]
        
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        explored_sequence: List[Tuple[int, int]] = []
        closed_set: Set[Tuple[int, int]] = set()

        while open_heap:
            current_cost, current_pos = heapq.heappop(open_heap)

            if current_pos in closed_set:
                continue
            
            closed_set.add(current_pos)
            explored_sequence.append(current_pos)

            if current_pos == goal:
                path: List[Tuple[int, int]] = []
                curr = current_pos
                while curr is not None:
                    path.append(curr)
                    curr = parent[curr]
                return path[::-1], explored_sequence

            for move in self.motion_vectors:
                neighbor_pos = (current_pos[0] + move[0], current_pos[1] + move[1])

                if not self.grid_map.is_valid_index(*neighbor_pos):
                    continue
                
                if self.grid_map.is_occupied(*neighbor_pos, threshold=self.threshold):
                    continue

                step_cost = float(np.linalg.norm(np.array(neighbor_pos) - np.array(current_pos)))
                tentative_g = current_cost + step_cost

                if neighbor_pos not in g_score or tentative_g < g_score[neighbor_pos]:
                    g_score[neighbor_pos] = tentative_g
                    parent[neighbor_pos] = current_pos
                    heapq.heappush(open_heap, (tentative_g, neighbor_pos))

        return None, explored_sequence
