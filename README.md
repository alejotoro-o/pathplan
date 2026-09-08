# PathPlan

![PyPI - License](https://img.shields.io/pypi/l/pathplan)
![PyPI - Version](https://img.shields.io/pypi/v/pathplan)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pathplan)
![PyPI Downloads](https://pepy.tech/badge/pathplan)

A lightweight, extensible Python library for 2D path planning and navigation. PathPlan provides a unified interface for implementing, comparing, and visualizing a wide array of classic and modern pathfinding algorithms.

## Features
- **Unified Interface**: All planners follow a consistent `BaseSolver` API.
- **Rich Algorithm Library**: Includes 27 different path planning strategies ranging from graph-search to sampling-based, reactive, metaheuristic, reinforcement learning, and multi-agent methods.
- **Standardized Mapping**: Uses a normalized `GridMap` (0.0 for free space, 1.0 for obstacles).
- **Visualization**: Built-in utilities for rendering search progress (explored nodes/edges) and final paths.

## Installation

Install the library directly from PyPI:

```bash
# Basic installation — includes all classic and multi-agent planners
pip install pathplan

# Install with plotting support
pip install pathplan[plot]

# Install with metaheuristic planners (GA, PSO, GWO, WOA)
pip install pathplan[metaheuristic]

# Install with reinforcement learning planners (Q-Learning, SARSA)
pip install pathplan[rl]

# Install everything
pip install pathplan[all]
```

Alternatively, for development or to use the latest source:

```bash
git clone https://github.com/alejotoro-o/pathplan.git
cd pathplan
pip install .[plot]
```

> **Note:** The base installation includes all classic and multi-agent planners.
> Metaheuristic planners require [`metaheuropt`](https://github.com/alejotoro-o/metaheuropt).
> RL planners require [`rlforge`](https://github.com/alejotoro-o/rlforge).
> Both are installed automatically with the corresponding extras.

## Quick Start

```python
import numpy as np
from pathplan.core import GridMap
from pathplan.classic import AStarPlanner
from pathplan.utils import Visualizer

# Create a 20x20 grid with some obstacles
data = np.zeros((20, 20))
data[5:15, 10] = 1.0  # Vertical wall

# Initialize map and planner
grid_map = GridMap(data)
planner = AStarPlanner(grid_map)

# Plan path
start, goal = (0, 0), (19, 19)
path, explored = planner.plan(start, goal)

# Visualize
viz = Visualizer(grid_map)
viz.plot_path(path, explored, title="A* Path Planning")
```

## Available Algorithms

### Classic Planners (`pathplan.classic`)

| Code | Full Name |
| :--- | :--- |
| `DijkstraPlanner` | Dijkstra's Algorithm |
| `AStarPlanner` | A* Search Algorithm |
| `BidirectionalAStarPlanner` | Bidirectional A* |
| `ThetaStarPlanner` | Theta* (Any-Angle A*) |
| `JPSPlanner` | Jump Point Search |
| `DStarLitePlanner` | D* Lite |
| `RRTPlanner` | Rapidly-exploring Random Tree |
| `RRTStarPlanner` | RRT* (Optimal RRT) |
| `InformedRRTStarPlanner` | Informed RRT* |
| `PRMPlanner` | Probabilistic Roadmap |
| `APFPlanner` | Artificial Potential Fields |
| `Bug2Planner` | Bug2 Algorithm |

### Metaheuristic Planners (`pathplan.metaheuristic`)

Requires the external [`metaheuropt`](https://github.com/alejotoro-o/metaheuropt) library.

| Code | Full Name |
| :--- | :--- |
| `GAPlanner` | Genetic Algorithm |
| `PSOPlanner` | Particle Swarm Optimization |
| `GWOPlanner` | Grey Wolf Optimizer |
| `WOAPlanner` | Whale Optimization Algorithm |

### RL Planners (`pathplan.rl`)

Requires the external [`rlforge`](https://github.com/alejotoro-o/rlforge) library.

| Code | Full Name |
| :--- | :--- |
| `QLearningPlanner` | Q-Learning |
| `SarsaPlanner` | SARSA |

### Multi-Agent Planners (`pathplan.multiagent`)

| Code | Full Name |
| :--- | :--- |
| `IndependentPlanner` | Independent Planning |
| `PrioritizedPlanner` | Prioritized Planning |
| `CBSPlanner` | Conflict-Based Search |
| `ECBSPlanner` | Enhanced Conflict-Based Search |
| `IDPlanner` | Independence Detection |
| `PBSPlanner` | Priority-Based Search |

Includes `MultiAgentVisualizer` for rendering multiple agent paths on a single grid.

## Testing

Run the suite of automated tests to verify planner integrity:
```bash
pytest -v
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
