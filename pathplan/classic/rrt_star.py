import random
import numpy as np
from typing import Tuple, List, Optional

from pathplan.core.base_solver import BaseSolver
from pathplan.core.map import GridMap
from pathplan.utils.geometry import edge_collision_free

class RRTStarPlanner(BaseSolver):
    """Asymptotically optimal sampling-based path planner (RRT*)."""

    def __init__(
        self,
        grid_map: GridMap,
        step_size: float = 2.0,
        goal_sample_rate: float = 0.2,
        goal_threshold: float = 3.0,
        neighbor_radius: float = 3.0,
        max_nodes: int = 3000,
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
        if dist == 0:
            return from_node
        direction = vec / dist
        new_point = np.array(from_node) + self.step_size * direction
        return (float(new_point[0]), float(new_point[1]))

    def plan(
        self, start: Tuple[float, float], goal: Tuple[float, float]
    ) -> Tuple[Optional[List[Tuple[float, float]]], List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
        
        # Edge Case: Start is already at or within threshold of goal
        if self._distance(start, goal) < self.goal_threshold:
            if edge_collision_free(start, goal, self.grid_map, threshold=self.threshold):
                if start == goal:
                    return [start], []
                return [start, goal], [(start, goal)]

        tree = {start: None}
        cost = {start: 0.0}
        nodes = [start]
        explored_edges = []

        for _ in range(self.max_nodes):
            if random.random() < self.goal_sample_rate:
                sample = goal
            else:
                sample = (
                    random.uniform(0, self.grid_map.height - 1),
                    random.uniform(0, self.grid_map.width - 1)
                )

            nearest = min(nodes, key=lambda n: self._distance(n, sample))
            new_node = self._steer(nearest, sample)

            if any(self._distance(new_node, n) < 1e-6 for n in nodes):
                continue

            if not edge_collision_free(nearest, new_node, self.grid_map, threshold=self.threshold):
                continue

            neighbors = [
                n for n in nodes 
                if self._distance(n, new_node) < self.neighbor_radius
                and edge_collision_free(n, new_node, self.grid_map, threshold=self.threshold)
            ]

            if not neighbors:
                parent = nearest
            else:
                parent = min(neighbors, key=lambda n: cost[n] + self._distance(n, new_node))

            tree[new_node] = parent
            cost[new_node] = cost[parent] + self._distance(parent, new_node)
            nodes.append(new_node)
            explored_edges.append((parent, new_node))

            for n in neighbors:
                new_cost = cost[new_node] + self._distance(new_node, n)
                if new_cost < cost[n] and edge_collision_free(new_node, n, self.grid_map, threshold=self.threshold):
                    tree[n] = new_node
                    cost[n] = new_cost
                    explored_edges.append((new_node, n))

            if self._distance(new_node, goal) < self.goal_threshold:
                if edge_collision_free(new_node, goal, self.grid_map, threshold=self.threshold):
                    path = [goal]
                    current = new_node
                    while current is not None:
                        path.append(current)
                        current = tree[current]
                    return path[::-1], explored_edges

        return None, explored_edges