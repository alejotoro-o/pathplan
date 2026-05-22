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

| Code / Abbreviation | Full Name | Original Reference |
| :--- | :--- | :--- |
| `DijkstraPlanner` | Dijkstra's Algorithm | [Dijkstra (1959)](https://dl.acm.org/doi/10.1145/368959.368993) |
| `AStarPlanner` | A* Search Algorithm | [Hart et al. (1968)](https://ieeexplore.ieee.org/document/4082128) |
| `BidirectionalAStarPlanner` | Bidirectional A* | [Pohl (1971)](https://dl.acm.org/doi/10.1145/321650.321651) |
| `ThetaStarPlanner` | Theta* (Any-Angle A*) | [Nash et al. (2007)](https://dl.acm.org/doi/10.5555/1283383.1283444) |
| `JPSPlanner` | Jump Point Search | [Harabor & Grastien (2011)](https://www.aaai.org/ocs/index.php/ICAPS/ICAPS11/paper/view/3432) |
| `DStarLitePlanner` | D* Lite | [Koenig & Likhachev (2002)](https://www.aaai.org/Papers/AAAI/2002/AAAI02-072.pdf) |
| `RRTPlanner` | Rapidly-exploring Random Tree | [LaValle (1998)](http://msl.cs.uiuc.edu/~lavalle/papers/Lav98c.pdf) |
| `RRTStarPlanner` | RRT* (Optimal RRT) | [Karaman & Frazzoli (2011)](https://journals.sagepub.org/doi/10.1177/0278364911406761) |
| `InformedRRTStarPlanner` | Informed RRT* | [Gammell et al. (2014)](https://ieeexplore.ieee.org/document/6942976) |
| `PRMPlanner` | Probabilistic Roadmap | [Kavraki et al. (1996)](https://ieeexplore.ieee.org/document/508439) |
| `APFPlanner` | Artificial Potential Fields | [Khatib (1986)](https://ieeexplore.ieee.org/document/4089857) |
| `Bug2Planner` | Bug2 Algorithm | [Lumelsky & Stepanov (1987)](https://ieeexplore.ieee.org/document/4100236) |

## Testing
Run the suite of automated tests to verify planner integrity:
```bash
pytest -v
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
