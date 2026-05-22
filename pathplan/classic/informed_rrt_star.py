import random
import numpy as np
from typing import Tuple, List, Optional

from pathplan.core.base_solver import BaseSolver
from pathplan.core.map import GridMap
from pathplan.utils.geometry import edge_collision_free

class InformedRRTStarPlanner(BaseSolver):
    """Informed RRT* path planner."""

    def __init__(
        self,
        grid_map: GridMap,
        step_size: float = 2.0,
        goal_sample_rate: float = 0.2,
        goal_threshold: float = 3.0,
        neighbor_radius: float = 5.0,
        max_nodes: int = 1000,
        occupancy_threshold: float = 0.5
    ):
        super().__init__(grid_map)
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.goal_threshold = goal_threshold
        self.neighbor_radius = neighbor_radius
        self.max_nodes = max_nodes
        self.threshold = occupancy_threshold

    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return float(np.linalg.norm(np.array(p1) - np.array(p2)))

    def _steer(self, from_node: Tuple[float, float], to_point: Tuple[float, float]) -> Tuple[float, float]:
        vec = np.array(to_point) - np.array(from_node)
        dist = np.linalg.norm(vec)
        if dist <= self.step_size:
            return to_point
        direction = vec / dist
        new_point = np.array(from_node) + self.step_size * direction
        return (float(new_point[0]), float(new_point[1]))

    def _sample(self, start: Tuple[float, float], goal: Tuple[float, float], c_best: float) -> Tuple[float, float]:
        if c_best < float('inf'):
            c_min = self._distance(start, goal)
            x_center = (np.array(start) + np.array(goal)) / 2.0
            
            # Rotation matrix from local to global
            a1 = (np.array(goal) - np.array(start)) / c_min
            id1_t = np.array([1.0, 0.0])
            M = np.outer(a1, id1_t)
            U, S, Vh = np.linalg.svd(M)
            C = U @ np.diag([1, np.linalg.det(U) * np.linalg.det(Vh)]) @ Vh

            # Radii of ellipse
            r1 = c_best / 2.0
            r2 = np.sqrt(c_best**2 - c_min**2) / 2.0
            L = np.diag([r1, r2])

            while True:
                x_ball = self._sample_unit_ball()
                x_rand = C @ L @ x_ball + x_center
                if 0 <= x_rand[0] < self.grid_map.height and 0 <= x_rand[1] < self.grid_map.width:
                    return (float(x_rand[0]), float(x_rand[1]))
        else:
            if random.random() < self.goal_sample_rate:
                return goal
            return (random.uniform(0, self.grid_map.height - 1), random.uniform(0, self.grid_map.width - 1))

    def _sample_unit_ball(self) -> np.ndarray:
        while True:
            x = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
            if np.linalg.norm(x) <= 1:
                return x

    def plan(
        self, start: Tuple[float, float], goal: Tuple[float, float]
    ) -> Tuple[Optional[List[Tuple[float, float]]], List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
        
        if self._distance(start, goal) < self.goal_threshold:
            if edge_collision_free(start, goal, self.grid_map, self.threshold):
                if start == goal:
                    return [start], []
                return [start, goal], [(start, goal)]
        
        tree = {start: None}
        cost = {start: 0.0}
        nodes = [start]
        explored_edges = []
        c_best = float('inf')
        best_path = None

        for _ in range(self.max_nodes):
            sample = self._sample(start, goal, c_best)
            nearest = min(nodes, key=lambda n: self._distance(n, sample))
            new_node = self._steer(nearest, sample)

            if self.grid_map.is_occupied(int(new_node[0]), int(new_node[1]), self.threshold):
                continue
            
            if not edge_collision_free(nearest, new_node, self.grid_map, self.threshold):
                continue

            neighbors = [n for n in nodes if self._distance(n, new_node) < self.neighbor_radius]
            
            # Find best parent
            parent = nearest
            min_cost = cost[nearest] + self._distance(nearest, new_node)
            for n in neighbors:
                if edge_collision_free(n, new_node, self.grid_map, self.threshold):
                    if cost[n] + self._distance(n, new_node) < min_cost:
                        parent = n
                        min_cost = cost[n] + self._distance(n, new_node)
            
            tree[new_node] = parent
            cost[new_node] = min_cost
            nodes.append(new_node)
            explored_edges.append((parent, new_node))

            # Rewire
            for n in neighbors:
                if edge_collision_free(new_node, n, self.grid_map, self.threshold):
                    if cost[new_node] + self._distance(new_node, n) < cost[n]:
                        tree[n] = new_node
                        cost[n] = cost[new_node] + self._distance(new_node, n)
                        explored_edges.append((new_node, n))

            # Check for path to goal
            if self._distance(new_node, goal) < self.goal_threshold:
                if edge_collision_free(new_node, goal, self.grid_map, self.threshold):
                    c_new = cost[new_node] + self._distance(new_node, goal)
                    if c_new < c_best:
                        c_best = c_new
                        path = [goal]
                        curr = new_node
                        while curr is not None:
                            path.append(curr)
                            curr = tree[curr]
                        best_path = path[::-1]

        return best_path, explored_edges
