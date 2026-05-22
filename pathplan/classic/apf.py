import numpy as np
from typing import Tuple, List, Optional

from pathplan.core.base_solver import BaseSolver
from pathplan.core.map import GridMap

class APFPlanner(BaseSolver):
    """Artificial Potential Field (APF) path planner."""

    def __init__(
        self,
        grid_map: GridMap,
        k_att: float = 1.0,
        k_rep: float = 100.0,
        rep_radius: float = 2.0,
        step_size: float = 0.5,
        max_iters: int = 1000,
        goal_threshold: float = 0.5,
        occupancy_threshold: float = 0.5
    ):
        super().__init__(grid_map)
        self.k_att = k_att
        self.k_rep = k_rep
        self.rep_radius = rep_radius
        self.step_size = step_size
        self.max_iters = max_iters
        self.goal_threshold = goal_threshold
        self.threshold = occupancy_threshold

    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return float(np.linalg.norm(np.array(p1) - np.array(p2)))

    def _get_potential_gradient(self, pos: Tuple[float, float], goal: Tuple[float, float]) -> np.ndarray:
        # Attractive force
        grad_att = self.k_att * (np.array(pos) - np.array(goal))
        
        # Repulsive force
        grad_rep = np.zeros(2)
        
        # We check nearby cells for obstacles to calculate repulsive force
        r_idx, c_idx = int(round(pos[0])), int(round(pos[1]))
        search_range = int(np.ceil(self.rep_radius))
        
        for dr in range(-search_range, search_range + 1):
            for dc in range(-search_range, search_range + 1):
                nr, nc = r_idx + dr, c_idx + dc
                if self.grid_map.is_valid_index(nr, nc) and self.grid_map.is_occupied(nr, nc, self.threshold):
                    obs_pos = np.array([nr, nc])
                    dist = np.linalg.norm(np.array(pos) - obs_pos)
                    if dist < self.rep_radius:
                        if dist < 0.1: dist = 0.1 # Avoid division by zero
                        grad_rep += self.k_rep * (1/self.rep_radius - 1/dist) * (1/dist**2) * (np.array(pos) - obs_pos) / dist
        
        return grad_att + grad_rep

    def plan(
        self, start: Tuple[float, float], goal: Tuple[float, float]
    ) -> Tuple[Optional[List[Tuple[float, float]]], List[Tuple[float, float]]]:
        
        if self._distance(start, goal) < self.goal_threshold:
            return [start], [start]
            
        path = [start]
        curr = np.array(start)
        explored = [start]

        for _ in range(self.max_iters):
            if self._distance(tuple(curr), goal) < self.goal_threshold:
                path.append(goal)
                return path, explored

            grad = self._get_potential_gradient(tuple(curr), goal)
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 0:
                grad = grad / grad_norm
            
            curr = curr - self.step_size * grad
            
            # Boundary check
            curr[0] = np.clip(curr[0], 0, self.grid_map.height - 1)
            curr[1] = np.clip(curr[1], 0, self.grid_map.width - 1)
            
            pos_tuple = (float(curr[0]), float(curr[1]))
            path.append(pos_tuple)
            explored.append(pos_tuple)
            
            # Local minima check (if we stop moving)
            if self._distance(path[-1], path[-2]) < self.step_size * 0.1:
                break

        return None, explored
