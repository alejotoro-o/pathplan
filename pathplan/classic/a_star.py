import heapq
import numpy as np
from typing import Tuple, List, Optional, Set, Dict

from pathplan.core.map import GridMap
from pathplan.core.base_solver import BaseSolver

class AStarNode:
    """Represents a node tracking state within the A* graph search space."""
    def __init__(self, position: Tuple[int, int], parent: Optional['AStarNode'] = None):
        self.position = position
        self.parent = parent
        self.g = 0.0
        self.h = 0.0
        self.f = 0.0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AStarNode):
            return False
        return self.position == other.position

    def __hash__(self) -> int:
        return hash(self.position)
    
    def __lt__(self, other: 'AStarNode') -> bool:
        # Necessary override hook to allow heapq structural processing
        return self.f < other.f


class AStarPlanner(BaseSolver):
    """Classic 8-directional A* planning solver."""
    
    def __init__(self, grid_map: GridMap, occupancy_threshold: float = 0.5):
        super().__init__(grid_map)
        self.threshold = occupancy_threshold
        # 8-Directional Movements
        self.motion_vectors = [
            (0, -1), (0, 1), (-1, 0), (1, 0),   # Orthogonal
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # Diagonal
        ]

    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Calculates Manhattan distance heuristic match to your template baseline."""
        return float(abs(goal[0] - pos[0]) + abs(goal[1] - pos[1]))

    def plan(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[Optional[List[Tuple[int, int]]], List[Tuple[int, int]]]:
        """
        Computes the shortest path using A*.
        
        Returns:
            - List of tuples containing coordinates mapping from start to goal.
            - List of tuples containing coordinates processed as explored space.
        """
        start_node = AStarNode(start)
        goal_node = AStarNode(goal)

        # Priority Queue storage tracker elements: (f_score, unique_node_instance)
        open_heap: List[AStarNode] = []
        heapq.heappush(open_heap, start_node)
        
        # Track maps for explicit membership validation checks
        open_dict: Dict[Tuple[int, int], AStarNode] = {start: start_node}
        closed_set: Set[Tuple[int, int]] = set()
        explored_sequence: List[Tuple[int, int]] = []

        while open_heap:
            current_node = heapq.heappop(open_heap)
            current_pos = current_node.position
            
            # Synchronize dictionaries trackers
            if current_pos in open_dict and open_dict[current_pos] == current_node:
                del open_dict[current_pos]

            closed_set.add(current_pos)
            explored_sequence.append(current_pos)

            # Check if destination path reached
            if current_pos == goal:
                path: List[Tuple[int, int]] = []
                curr = current_node
                while curr is not None:
                    path.append(curr.position)
                    curr = curr.parent
                return path[::-1], explored_sequence

            # Expand neighbors
            for move in self.motion_vectors:
                neighbor_pos = (current_pos[0] + move[0], current_pos[1] + move[1])

                if not self.grid_map.is_valid_index(*neighbor_pos):
                    continue
                
                if self.grid_map.is_occupied(*neighbor_pos, threshold=self.threshold):
                    continue

                if neighbor_pos in closed_set:
                    continue

                # Cost calculations match your baseline vector norms
                step_cost = float(np.linalg.norm(np.array(neighbor_pos) - np.array(current_pos)))
                g_tentative = current_node.g + step_cost

                child = AStarNode(neighbor_pos, current_node)
                child.g = g_tentative
                child.h = self._heuristic(neighbor_pos, goal)
                child.f = child.g + child.h

                if neighbor_pos in open_dict:
                    if child.g >= open_dict[neighbor_pos].g:
                        continue
                    # Node optimization sequence modification adjustment
                    open_dict[neighbor_pos].g = child.g
                    open_dict[neighbor_pos].f = child.f
                    open_dict[neighbor_pos].parent = current_node
                    heapq.heapify(open_heap) # Force heap restructuring
                else:
                    open_dict[neighbor_pos] = child
                    heapq.heappush(open_heap, child)

        return None, explored_sequence