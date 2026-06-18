---
name: metaheuropt-path-planning
description: Workflow for implementing path planning algorithms using the metaheuropt optimization library. Use when developing new planners for PathPlan that rely on metaheuristic methods (PSO, GA, GWO, etc.) to optimize waypoint coordinates.
---

# Metaheuristic Path Planning with metaheuropt

## Overview
This skill provides a specialized workflow for implementing path planning algorithms using the `metaheuropt` optimization library within the `pathplan` framework. It focuses on transforming path planning into a continuous optimization problem. The documentation can be found in this [repository](https://github.com/alejotoro-o/metaheuropt).

## Imports
```python
from metaheuropt.core import Optimizer
from metaheuropt.solvers import GA, PSO, GWO  # or any of the 14 solvers
```

## Implementation Workflow

### 0. Edge Case: Start == Goal
If `start == goal`, return `[start]` immediately — no optimization needed.

### 1. Problem Representation
Represent a path of $N$ intermediate waypoints as a 1D vector $X$ of size $2N$.
$X = [r_1, c_1, r_2, c_2, \dots, r_N, c_N]$
The full path is: `[start] + waypoints + [goal]`.

### 2. Objective Function (Fitness)
Implement an objective function that `metaheuropt` can minimize:
- **Total Distance**: Calculate the Euclidean distance between all consecutive waypoints (including start and goal).
- **Collision Penalty**: For each segment $(p_i, p_{i+1})$, use `pathplan.utils.geometry.edge_collision_free`. Add a large penalty (e.g., $10^6$) for each colliding segment.

See [references/fitness_guidance.md](references/fitness_guidance.md) for a code template.

### 3. Solver Setup
```python
bounds = (np.zeros(dim), np.array([H, W] * num_waypoints))  # H=grid_map.height, W=grid_map.width
solver = GA(bounds, pop_size=50, max_iter=100)
# Or: PSO(bounds, pop_size=50, max_iter=100)
# Or: GWO(bounds, pop_size=50, max_iter=100)
```
Each solver's `.name` attribute matches the `get_overall_best()` key: `"GA"`, `"PSO"`, `"GWO"`, etc.
For solver-specific hyperparameters (e.g., GA `pc`, `eta_c`, `eta_m`; PSO `w_max`, `w_min`, `c1`, `c2`), see the library source or docs.

### 4. Running Optimization
```python
optimizer = Optimizer(solver, obj_func, num_runs=1)
optimizer.run(save_results=False)           # save_results=False avoids writing files
best_run = optimizer.get_overall_best("GA") # key must match solver.name
best_solution = best_run["solution"]        # ndarray of shape (2*N,)
best_fitness = best_run["fitness"]          # scalar
```

### 5. Post-Processing & Path Assembly
```python
if best_fitness >= penalty:      # collision detected → invalid path
    return None, [tuple(w) for w in best_solution.reshape(-1, 2)]

waypoints = best_solution.reshape(-1, 2)      # reshape back to (N, 2)
path = [start] + [tuple(w) for w in waypoints] + [goal]
explored = [tuple(w) for w in waypoints]
return path, explored
```

### 6. PathPlan Integration
Wrap the optimization logic in a class inheriting from `pathplan.core.BaseSolver`. See `pathplan/metaheuristic/ga.py` for a complete reference implementation.

## Available Solver Import Paths
All solvers are accessible from `metaheuropt.solvers`:

| Import | Algorithm | Constructor |
|---|---|---|
| `GA` | Genetic Algorithm | `GA(bounds, pop_size, max_iter, pc, eta_c, eta_m)` |
| `PSO` | Particle Swarm Optimization | `PSO(bounds, pop_size, max_iter, w_max, w_min, c1, c2)` |
| `GWO` | Grey Wolf Optimizer | `GWO(bounds, pop_size, max_iter)` |
| `DE` | Differential Evolution | (see library) |
| `ABC` | Artificial Bee Colony | (see library) |
| `CMAES` | CMA-ES | (see library) |
| `SA` | Simulated Annealing | (see library) |
| `WOA` | Whale Optimization Algorithm | (see library) |
| `JADE` | Adaptive DE with Archive | (see library) |

Each solver's `__init__` accepts `stop_patience` for early stopping.

## Solver Selection
- **PSO**: Fast convergence, good for simple maps.
- **GA**: Diversity in complex maps with many local minima.
- **GWO**: Balanced exploration/exploitation, few hyperparameters.
