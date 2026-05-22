import heapq
import numpy as np
from typing import Tuple, List, Optional, Set, Dict

from pathplan.core.map import GridMap
from pathplan.core.base_solver import BaseSolver

class JPSPlanner(BaseSolver):
    """Jump Point Search (JPS) planning solver."""
    
    def __init__(self, grid_map: GridMap, occupancy_threshold: float = 0.5):
        super().__init__(grid_map)
        self.threshold = occupancy_threshold

    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        dx = abs(pos[0] - goal[0])
        dy = abs(pos[1] - goal[1])
        return float(max(dx, dy) + (np.sqrt(2) - 1) * min(dx, dy))

    def _is_occupied(self, pos: Tuple[int, int]) -> bool:
        return self.grid_map.is_occupied(pos[0], pos[1], self.threshold)

    def _get_neighbors(self, pos: Tuple[int, int], parent: Optional[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if parent is None:
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0: continue
                    n = (pos[0] + dx, pos[1] + dy)
                    if self.grid_map.is_valid_index(*n) and not self._is_occupied(n):
                        neighbors.append(n)
            return neighbors

        px, py = parent
        x, y = pos
        dx = (x - px) // max(1, abs(x - px))
        dy = (y - py) // max(1, abs(y - py))

        neighbors = []
        if dx != 0 and dy != 0:
            if not self._is_occupied((x + dx, y + dy)): neighbors.append((x + dx, y + dy))
            if not self._is_occupied((x + dx, y)): neighbors.append((x + dx, y))
            if not self._is_occupied((x, y + dy)): neighbors.append((x, y + dy))
            if self._is_occupied((x - dx, y)) and not self._is_occupied((x - dx, y + dy)): neighbors.append((x - dx, y + dy))
            if self._is_occupied((x, y - dy)) and not self._is_occupied((x + dx, y - dy)): neighbors.append((x + dx, y - dy))
        elif dx != 0:
            if not self._is_occupied((x + dx, y)):
                neighbors.append((x + dx, y))
                if self._is_occupied((x, y + 1)) and not self._is_occupied((x + dx, y + 1)): neighbors.append((x + dx, y + 1))
                if self._is_occupied((x, y - 1)) and not self._is_occupied((x + dx, y - 1)): neighbors.append((x + dx, y - 1))
        else:
            if not self._is_occupied((x, y + dy)):
                neighbors.append((x, y + dy))
                if self._is_occupied((x + 1, y)) and not self._is_occupied((x + 1, y + dy)): neighbors.append((x + 1, y + dy))
                if self._is_occupied((x - 1, y)) and not self._is_occupied((x - 1, y + dy)): neighbors.append((x - 1, y + dy))
        
        return [n for n in neighbors if self.grid_map.is_valid_index(*n)]

    def _jump(self, pos: Tuple[int, int], dx: int, dy: int, goal: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        nx, ny = pos[0] + dx, pos[1] + dy
        if not self.grid_map.is_valid_index(nx, ny) or self._is_occupied((nx, ny)):
            return None
        
        if (nx, ny) == goal:
            return (nx, ny)

        # Forced neighbors
        if dx != 0 and dy != 0:
            if (self.grid_map.is_valid_index(nx - dx, ny) and self._is_occupied((nx - dx, ny)) and not self._is_occupied((nx - dx, ny + dy))) or \
               (self.grid_map.is_valid_index(nx, ny - dy) and self._is_occupied((nx, ny - dy)) and not self._is_occupied((nx + dx, ny - dy))):
                return (nx, ny)
            if self._jump((nx, ny), dx, 0, goal) or self._jump((nx, ny), 0, dy, goal):
                return (nx, ny)
        elif dx != 0:
            if (self.grid_map.is_valid_index(nx, ny + 1) and self._is_occupied((nx, ny + 1)) and not self._is_occupied((nx + dx, ny + 1))) or \
               (self.grid_map.is_valid_index(nx, ny - 1) and self._is_occupied((nx, ny - 1)) and not self._is_occupied((nx + dx, ny - 1))):
                return (nx, ny)
        else:
            if (self.grid_map.is_valid_index(nx + 1, ny) and self._is_occupied((nx + 1, ny)) and not self._is_occupied((nx + 1, ny + dy))) or \
               (self.grid_map.is_valid_index(nx - 1, ny) and self._is_occupied((nx - 1, ny)) and not self._is_occupied((nx - 1, ny + dy))):
                return (nx, ny)

        return self._jump((nx, ny), dx, dy, goal)

    def plan(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[Optional[List[Tuple[int, int]]], List[Tuple[int, int]]]:
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))

        open_heap = [(self._heuristic(start, goal), start)]
        g_score = {start: 0.0}
        parent = {start: None}
        explored = []

        while open_heap:
            _, current = heapq.heappop(open_heap)
            explored.append(current)

            if current == goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = parent[current]
                return path[::-1], explored

            for neighbor in self._get_neighbors(current, parent[current]):
                dx = neighbor[0] - current[0]
                dy = neighbor[1] - current[1]
                jump_point = self._jump(current, dx, dy, goal)

                if jump_point and jump_point not in explored:
                    dist = np.linalg.norm(np.array(jump_point) - np.array(current))
                    new_g = g_score[current] + dist
                    if jump_point not in g_score or new_g < g_score[jump_point]:
                        g_score[jump_point] = new_g
                        parent[jump_point] = current
                        f = new_g + self._heuristic(jump_point, goal)
                        heapq.heappush(open_heap, (f, jump_point))

        return None, explored
