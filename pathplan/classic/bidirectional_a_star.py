import heapq
import numpy as np
from typing import Tuple, List, Optional, Set, Dict

from pathplan.core.map import GridMap
from pathplan.core.base_solver import BaseSolver

class BidirectionalAStarPlanner(BaseSolver):
    """Bidirectional A* planning solver."""
    
    def __init__(self, grid_map: GridMap, occupancy_threshold: float = 0.5):
        super().__init__(grid_map)
        self.threshold = occupancy_threshold
        self.motion_vectors = [
            (0, -1), (0, 1), (-1, 0), (1, 0),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]

    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        return float(np.linalg.norm(np.array(pos) - np.array(goal)))

    def plan(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[Optional[List[Tuple[int, int]]], List[Tuple[int, int]]]:
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))

        if start == goal:
            return [start], [start]

        # Forward search
        open_f = [(self._heuristic(start, goal), start)]
        g_f = {start: 0.0}
        parent_f = {start: None}
        closed_f = set()

        # Backward search
        open_b = [(self._heuristic(goal, start), goal)]
        g_b = {goal: 0.0}
        parent_b = {goal: None}
        closed_b = set()

        explored_sequence = []

        while open_f and open_b:
            # Expand forward
            _, curr_f = heapq.heappop(open_f)
            if curr_f not in closed_f:
                closed_f.add(curr_f)
                explored_sequence.append(curr_f)

                if curr_f in closed_b:
                    return self._reconstruct_path(parent_f, parent_b, curr_f), explored_sequence

                for move in self.motion_vectors:
                    neighbor = (curr_f[0] + move[0], curr_f[1] + move[1])
                    if self.grid_map.is_valid_index(*neighbor) and not self.grid_map.is_occupied(*neighbor, self.threshold):
                        new_g = g_f[curr_f] + np.linalg.norm(move)
                        if neighbor not in g_f or new_g < g_f[neighbor]:
                            g_f[neighbor] = new_g
                            parent_f[neighbor] = curr_f
                            heapq.heappush(open_f, (new_g + self._heuristic(neighbor, goal), neighbor))

            # Expand backward
            _, curr_b = heapq.heappop(open_b)
            if curr_b not in closed_b:
                closed_b.add(curr_b)
                explored_sequence.append(curr_b)

                if curr_b in closed_f:
                    return self._reconstruct_path(parent_f, parent_b, curr_b), explored_sequence

                for move in self.motion_vectors:
                    neighbor = (curr_b[0] + move[0], curr_b[1] + move[1])
                    if self.grid_map.is_valid_index(*neighbor) and not self.grid_map.is_occupied(*neighbor, self.threshold):
                        new_g = g_b[curr_b] + np.linalg.norm(move)
                        if neighbor not in g_b or new_g < g_b[neighbor]:
                            g_b[neighbor] = new_g
                            parent_b[neighbor] = curr_b
                            heapq.heappush(open_b, (new_g + self._heuristic(neighbor, start), neighbor))

        return None, explored_sequence

    def _reconstruct_path(self, parent_f: Dict, parent_b: Dict, meeting_point: Tuple[int, int]) -> List[Tuple[int, int]]:
        path_f = []
        curr = meeting_point
        while curr is not None:
            path_f.append(curr)
            curr = parent_f[curr]
        path_f = path_f[::-1]

        path_b = []
        curr = parent_b[meeting_point]
        while curr is not None:
            path_b.append(curr)
            curr = parent_b[curr]
        
        return path_f + path_b
