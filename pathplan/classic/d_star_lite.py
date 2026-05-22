import heapq
import numpy as np
(0, -1), (0, 1), (-1, 0), (1, 0),
(-1, -1), (-1, 1), (1, -1), (1, 1)
from typing import Tuple, List, Optional, Set, Dict

from pathplan.core.map import GridMap
from pathplan.core.base_solver import BaseSolver

class DStarLitePlanner(BaseSolver):
    """Simplified D* Lite planning solver."""
    
    def __init__(self, grid_map: GridMap, occupancy_threshold: float = 0.5):
        super().__init__(grid_map)
        self.threshold = occupancy_threshold
        self.motion_vectors = [
            (0, -1), (0, 1), (-1, 0), (1, 0),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]
        self.km = 0.0
        self.g = {}
        self.rhs = {}
        self.U = []

    def _heuristic(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        return float(np.linalg.norm(np.array(p1) - np.array(p2)))

    def _calculate_key(self, s: Tuple[int, int], start: Tuple[int, int]) -> Tuple[float, float]:
        min_g_rhs = min(self.g.get(s, float('inf')), self.rhs.get(s, float('inf')))
        return (min_g_rhs + self._heuristic(s, start) + self.km, min_g_rhs)

    def _update_vertex(self, s: Tuple[int, int], start: Tuple[int, int], goal: Tuple[int, int]):
        if s != goal:
            min_rhs = float('inf')
            for move in self.motion_vectors:
                s_prime = (s[0] + move[0], s[1] + move[1])
                if self.grid_map.is_valid_index(*s_prime) and not self.grid_map.is_occupied(*s_prime, self.threshold):
                    cost = np.linalg.norm(move)
                    min_rhs = min(min_rhs, cost + self.g.get(s_prime, float('inf')))
            self.rhs[s] = min_rhs
        
        # Remove s from U if it exists
        self.U = [item for item in self.U if item[1] != s]
        heapq.heapify(self.U)

        if self.g.get(s, float('inf')) != self.rhs.get(s, float('inf')):
            heapq.heappush(self.U, (self._calculate_key(s, start), s))

    def plan(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[Optional[List[Tuple[int, int]]], List[Tuple[int, int]]]:
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        
        self.U = []
        self.km = 0.0
        self.g = {}
        self.rhs = {goal: 0.0}
        heapq.heappush(self.U, (self._calculate_key(goal, start), goal))
        
        explored = []

        while self.U and (self.U[0][0] < self._calculate_key(start, start) or self.rhs.get(start, float('inf')) != self.g.get(start, float('inf'))):
            k_old, u = heapq.heappop(self.U)
            explored.append(u)
            
            if k_old < self._calculate_key(u, start):
                heapq.heappush(self.U, (self._calculate_key(u, start), u))
            elif self.g.get(u, float('inf')) > self.rhs.get(u, float('inf')):
                self.g[u] = self.rhs[u]
                for move in self.motion_vectors:
                    s = (u[0] + move[0], u[1] + move[1])
                    if self.grid_map.is_valid_index(*s) and not self.grid_map.is_occupied(*s, self.threshold):
                        self._update_vertex(s, start, goal)
            else:
                self.g[u] = float('inf')
                self._update_vertex(u, start, goal)
                for move in self.motion_vectors:
                    s = (u[0] + move[0], u[1] + move[1])
                    if self.grid_map.is_valid_index(*s) and not self.grid_map.is_occupied(*s, self.threshold):
                        self._update_vertex(s, start, goal)

        if self.rhs.get(start, float('inf')) == float('inf'):
            return None, explored

        # Reconstruct path
        path = [start]
        curr = start
        while curr != goal:
            best_s = None
            min_cost = float('inf')
            for move in self.motion_vectors:
                s = (curr[0] + move[0], curr[1] + move[1])
                if self.grid_map.is_valid_index(*s) and not self.grid_map.is_occupied(*s, self.threshold):
                    cost = np.linalg.norm(move) + self.g.get(s, float('inf'))
                    if cost < min_cost:
                        min_cost = cost
                        best_s = s
            if best_s is None or best_s == curr:
                break
            curr = best_s
            path.append(curr)
        
        return path, explored
