# PathPlan

![PyPI - License](https://img.shields.io/pypi/l/pathplan)
![PyPI - Version](https://img.shields.io/pypi/v/pathplan)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pathplan)
![PyPI Downloads](https://pepy.tech/badge/pathplan)

A lightweight, extensible Python library for 2D path planning and navigation. PathPlan provides a unified interface for implementing, comparing, and visualizing a wide array of classic and modern pathfinding algorithms.

## Features
- **Unified Interface**: All planners follow a consistent `BaseSolver` API.
- **Rich Algorithm Library**: Includes 12 different path planning strategies ranging from graph-search to sampling-based and reactive methods.
- **Standardized Mapping**: Uses a normalized `GridMap` (0.0 for free space, 1.0 for obstacles).
- **Visualization**: Built-in utilities for rendering search progress (explored nodes/edges) and final paths.

## Installation

Install the library directly from PyPI:

```bash
# Basic installation
pip install pathplan

# Install with plotting support
pip install pathplan[plot]
```

Alternatively, for development or to use the latest source:

```bash
git clone https://github.com/alejotoro-o/pathplan.git
cd pathplan
pip install .[plot]
```

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

## Testing

Run the suite of automated tests to verify planner integrity:
```bash
pytest -v
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
