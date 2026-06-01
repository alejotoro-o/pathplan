import numpy as np
from typing import Tuple, List, Optional, Any

from pathplan.core.base_solver import BaseSolver
from pathplan.core.map import GridMap
from pathplan.utils.geometry import edge_collision_free

# Importing metaheuropt components
try:
    from metaheuropt.core.optimizer import Optimizer
    from metaheuropt.solvers.ga import GA
except ImportError:
    raise ImportError(
        "metaheuropt library is required for GAPlanner. "
        "Please ensure it is installed and available in your PYTHONPATH."
    )

class GAPlanner(BaseSolver):
    """
    Path Planning using Genetic Algorithm (GA) from metaheuropt library.
    
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
        occupancy_threshold: float = 0.5
    ):
        """
        Args:
            grid_map: The occupancy grid map.
            num_waypoints: Number of intermediate points to optimize.
            pop_size: Population size for GA.
            max_iter: Maximum number of generations.
            collision_penalty: Penalty added to fitness for each colliding segment.
            occupancy_threshold: Threshold for considering a cell occupied.
        """
        super().__init__(grid_map)
        self.num_waypoints = num_waypoints
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.penalty = collision_penalty
        self.threshold = occupancy_threshold

    def _create_fitness_function(self, start: Tuple[float, float], goal: Tuple[float, float]):
        """Creates the objective function for metaheuropt to minimize."""
        def fitness(X):
            # Reshape X into a list of waypoints
            waypoints = X.reshape(-1, 2)
            full_path = [start] + [tuple(w) for w in waypoints] + [goal]
            
            total_dist = 0.0
            collision_penalty = 0.0
            
            for i in range(len(full_path) - 1):
                p1, p2 = full_path[i], full_path[i+1]
                
                # 1. Distance Cost
                total_dist += np.linalg.norm(np.array(p1) - np.array(p2))
                
                # 2. Collision Penalty
                if not edge_collision_free(p1, p2, self.grid_map, threshold=self.threshold):
                    collision_penalty += self.penalty
                    
            return total_dist + collision_penalty
        return fitness

    def plan(self, start: Tuple[float, float], goal: Tuple[float, float]) -> Tuple[Optional[List[Tuple[float, float]]], List[Tuple[float, float]]]:
        """
        Executes GA optimization to find a path.
        """
        # Edge Case: Start == Goal
        if start == goal:
            return [start], []

        # 1. Problem Representation (Bounds)
        # We optimize num_waypoints * 2 variables (r, c for each waypoint)
        dim = self.num_waypoints * 2
        lb = np.zeros(dim)
        ub = np.zeros(dim)
        for i in range(0, dim, 2):
            lb[i], lb[i+1] = 0, 0
            ub[i], ub[i+1] = self.grid_map.height - 1, self.grid_map.width - 1
        
        bounds = (lb, ub)

        # 2. Solver and Optimizer Setup
        ga_solver = GA(bounds, pop_size=self.pop_size, max_iter=self.max_iter)
        obj_func = self._create_fitness_function(start, goal)
        
        # We run only 1 run for planning, but Optimizer expects num_runs
        optimizer = Optimizer(ga_solver, obj_func, num_runs=1)
        
        # 3. Run Optimization (Supressing output if possible, but following standard for now)
        optimizer.run(save_results=False)
        
        # 4. Extract Best Result
        best_run = optimizer.get_overall_best("GA")
        if best_run is None:
            return None, []
        
        best_solution = best_run["solution"]
        best_fitness = best_run["fitness"]
        
        # Check if the path is collision-free (fitness should not have penalty)
        # If best_fitness > penalty, it means there is at least one collision
        if best_fitness >= self.penalty:
            # We still return the path but the user should know it might collide
            # However, BaseSolver contract usually implies a valid path or None
            # Let's be strict: if it collides, return None for path
            # But we can return the waypoints as explored data
            waypoints = best_solution.reshape(-1, 2)
            explored = [tuple(w) for w in waypoints]
            return None, explored

        # Construct final path
        waypoints = best_solution.reshape(-1, 2)
        path = [start] + [tuple(w) for w in waypoints] + [goal]
        
        # Explored in this context are the waypoints of the best individual
        explored = [tuple(w) for w in waypoints]
        
        return path, explored
