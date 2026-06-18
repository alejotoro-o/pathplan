# Fitness Function Implementation for Path Planning

To use `metaheuropt`, you need to provide a function $f(X) \to \mathbb{R}$.

```python
import numpy as np
from pathplan.utils.geometry import edge_collision_free

def create_fitness_function(grid_map, start, goal, penalty=1e6, threshold=0.5):
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
            if not edge_collision_free(p1, p2, grid_map, threshold=threshold):
                collision_penalty += penalty
                
        return total_dist + collision_penalty
    return fitness
```

## Tips for Better Convergence
1. **Normalization**: Consider normalizing coordinates to $[0, 1]$ if the solver performs better with small ranges.
2. **Smoothness**: Add a penalty for sharp turns to produce more "robotic" paths.
3. **Static Obstacles**: If the map is static, you can pre-compute a distance transform to provide a "gradient" away from obstacles.

## Notes
- `edge_collision_free(p1, p2, grid_map, threshold=...)` — the `threshold` parameter must match the planner's `occupancy_threshold` for consistency.
