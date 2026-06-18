import numpy as np
from typing import Tuple, List, Optional, Any

from pathplan.core.base_solver import BaseSolver
from pathplan.core.map import GridMap
from pathplan.utils.geometry import edge_collision_free

try:
    from metaheuropt.core.optimizer import Optimizer
    from metaheuropt.solvers import PSO
except ImportError:
    raise ImportError(
        "metaheuropt library is required for PSOPlanner. "
        "Please ensure it is installed and available in your PYTHONPATH."
    )

class PSOPlanner(BaseSolver):
    """
    Path Planning using Particle Swarm Optimization (PSO) from metaheuropt library.
    
    This planner transforms the path planning problem into a continuous 
    optimization problem where a set of intermediate waypoints are optimized 
    to minimize distance while avoiding obstacles.
    """

    def __init__(
        self, 
        grid_map: GridMap, 
        num_waypoints: int = 5,
        pop_size: int = 50,
        max_iter: int = 100,
        collision_penalty: float = 1e6,
        occupancy_threshold: float = 0.5,
        w_max: float = 0.95,
        w_min: float = 0.35,
        c1: float = 1.4,
        c2: float = 1.4
    ):
        super().__init__(grid_map)
        self.num_waypoints = num_waypoints
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.penalty = collision_penalty
        self.threshold = occupancy_threshold
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2

    def _create_fitness_function(self, start: Tuple[float, float], goal: Tuple[float, float]):
        def fitness(X):
            waypoints = X.reshape(-1, 2)
            full_path = [start] + [tuple(w) for w in waypoints] + [goal]
            
            total_dist = 0.0
            collision_penalty = 0.0
            
            for i in range(len(full_path) - 1):
                p1, p2 = full_path[i], full_path[i+1]
                
                total_dist += np.linalg.norm(np.array(p1) - np.array(p2))
                
                if not edge_collision_free(p1, p2, self.grid_map, threshold=self.threshold):
                    collision_penalty += self.penalty
                    
            return total_dist + collision_penalty
        return fitness

    def plan(self, start: Tuple[float, float], goal: Tuple[float, float]) -> Tuple[Optional[List[Tuple[float, float]]], List[Tuple[float, float]]]:
        if start == goal:
            return [start], []

        dim = self.num_waypoints * 2
        lb = np.zeros(dim)
        ub = np.zeros(dim)
        for i in range(0, dim, 2):
            lb[i], lb[i+1] = 0, 0
            ub[i], ub[i+1] = self.grid_map.height - 1, self.grid_map.width - 1
        
        bounds = (lb, ub)

        pso_solver = PSO(bounds, pop_size=self.pop_size, max_iter=self.max_iter,
                         w_max=self.w_max, w_min=self.w_min, c1=self.c1, c2=self.c2)
        obj_func = self._create_fitness_function(start, goal)
        
        optimizer = Optimizer(pso_solver, obj_func, num_runs=1)
        optimizer.run(save_results=False)
        
        best_run = optimizer.get_overall_best("PSO")
        if best_run is None:
            return None, []
        
        best_solution = best_run["solution"]
        best_fitness = best_run["fitness"]
        
        if best_fitness >= self.penalty:
            waypoints = best_solution.reshape(-1, 2)
            explored = [tuple(w) for w in waypoints]
            return None, explored

        waypoints = best_solution.reshape(-1, 2)
        path = [start] + [tuple(w) for w in waypoints] + [goal]
        explored = [tuple(w) for w in waypoints]
        
        return path, explored
