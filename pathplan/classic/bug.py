import numpy as np
from typing import Tuple, List, Optional

from pathplan.core.base_solver import BaseSolver
from pathplan.core.map import GridMap
from pathplan.utils.geometry import edge_collision_free

class Bug2Planner(BaseSolver):
    """Bug2 path planning solver."""

    def __init__(
        self,
        grid_map: GridMap,
        step_size: float = 1.0,
        max_iters: int = 5000,
        goal_threshold: float = 1.0,
        occupancy_threshold: float = 0.5
    ):
        super().__init__(grid_map)
        self.step_size = step_size
        self.max_iters = max_iters
        self.goal_threshold = goal_threshold
        self.threshold = occupancy_threshold

    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return float(np.linalg.norm(np.array(p1) - np.array(p2)))

    def _is_on_m_line(self, pos: Tuple[float, float], start: Tuple[float, float], goal: Tuple[float, float]) -> bool:
        # Distance from point to line (m-line)
        p1 = np.array(start)
        p2 = np.array(goal)
        p3 = np.array(pos)
        
        if np.array_equal(p1, p2): return True
        
        # Padding to 3D for np.cross to avoid deprecation warning
        p1_3d = np.append(p1, 0)
        p2_3d = np.append(p2, 0)
        p3_3d = np.append(p3, 0)
        
        dist = np.linalg.norm(np.cross(p2_3d-p1_3d, p1_3d-p3_3d)) / np.linalg.norm(p2-p1)
        return dist < 0.5 # Tolerance for grid-based m-line

    def plan(
        self, start: Tuple[float, float], goal: Tuple[float, float]
    ) -> Tuple[Optional[List[Tuple[float, float]]], List[Tuple[float, float]]]:
        
        if self._distance(start, goal) < self.goal_threshold:
            return [start], [start]

        path = [start]
        curr = np.array(start)
        explored = [start]
        
        state = "GO_TO_GOAL"
        hit_point = None
        
        # 8 directions for wall following
        directions = [
            (0, 1), (1, 1), (1, 0), (1, -1),
            (0, -1), (-1, -1), (-1, 0), (-1, 1)
        ]

        for _ in range(self.max_iters):
            if self._distance(tuple(curr), goal) < self.goal_threshold:
                path.append(goal)
                return path, explored

            if state == "GO_TO_GOAL":
                # Move towards goal
                vec = np.array(goal) - curr
                direction = vec / np.linalg.norm(vec)
                next_pos = curr + self.step_size * direction
                
                if self.grid_map.is_occupied(int(round(next_pos[0])), int(round(next_pos[1])), self.threshold):
                    state = "WALL_FOLLOW"
                    hit_point = tuple(curr)
                else:
                    curr = next_pos
                    path.append(tuple(curr))
                    explored.append(tuple(curr))

            elif state == "WALL_FOLLOW":
                # Follow obstacle boundary
                # Try to move around the obstacle
                moved = False
                
                # Simple wall following logic (left-hand)
                # Find current orientation relative to obstacle
                # This is a simplified version for 2D grids
                best_n = None
                min_dist_to_goal = float('inf')
                
                for dx, dy in directions:
                    n = (int(round(curr[0] + dx)), int(round(curr[1] + dy)))
                    if self.grid_map.is_valid_index(*n) and not self.grid_map.is_occupied(*n, self.threshold):
                        # Candidate for moving
                        dist = self._distance(n, goal)
                        # We want to move along the wall, so we look for neighbors that are next to occupied cells
                        is_wall_neighbor = False
                        for ddx, ddy in directions:
                            nn = (n[0] + ddx, n[1] + ddy)
                            if self.grid_map.is_valid_index(*nn) and self.grid_map.is_occupied(*nn, self.threshold):
                                is_wall_neighbor = True
                                break
                        
                        if is_wall_neighbor:
                            # Prefer points that maintain wall contact but don't loop back to hit_point too early
                            if best_n is None or dist < min_dist_to_goal:
                                # Avoid immediate backtrack
                                if self._distance(n, path[-2]) > 0.5:
                                    best_n = n
                                    min_dist_to_goal = dist
                
                if best_n:
                    curr = np.array(best_n, dtype=float)
                    path.append(tuple(curr))
                    explored.append(tuple(curr))
                    
                    # Check if we can resume GO_TO_GOAL
                    if self._is_on_m_line(tuple(curr), start, goal) and self._distance(tuple(curr), goal) < self._distance(hit_point, goal) - 1.0:
                        state = "GO_TO_GOAL"
                else:
                    # Trapped or circular obstacle
                    break

        return None, explored
