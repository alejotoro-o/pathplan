# PathPlan — Agent Guide

## Project structure

```
pathplan/           # Library package
  core/             # BaseSolver (ABC), GridMap
  classic/          # 12 planners (Dijkstra, A*, RRT, JPS, PRM, Bug2, APF, etc.)
  metaheuristic/    # GA planner (requires metaheuropt)
  utils/            # geometry (edge_collision_free), Visualizer
tests/              # pytest, conftest.py with fixtures
examples/           # Jupyter notebooks
```

## Adding a planner

1. Create a class in `pathplan/classic/` (or `pathplan/metaheuristic/`) extending `BaseSolver`.
2. Implement `plan(start, goal) -> Tuple[Optional[List[Point]], Any]`:
   - `path`: coordinates from start to goal, or `None` if unreachable.
   - `explored`: diagnostic data (nodes for graph search, edges for sampling search).
3. Register in the subpackage's `__init__.py` and add to `__all__`.
4. For classic planners: add the class to `CLASSIC_PLANNERS` in `tests/test_classic.py`.
5. Must handle: `start == goal` → `[start]`, unreachable → `(None, explored)`.

## Point type conventions

| Planner type | Point type |
|---|---|
| Graph-search (Dijkstra, A*, Theta*, JPS, D* Lite, Bug2, APF) | `Tuple[int, int]` |
| Sampling-based (RRT, RRT*, Informed RRT*, PRM) | `Tuple[float, float]` |
| Metaheuristic (GA) | `Tuple[float, float]` |

Use `pathplan.utils.geometry.edge_collision_free(p1, p2, grid_map)` for segment collision checks.

## Commands

```bash
pip install -e .          # dev install (numpy only)
pip install -e .[plot]    # + matplotlib
pip install -e .[test]    # + pytest
pytest -v                 # run all tests
pytest tests/test_classic.py  # single file
```

## Dependencies

- **Mandatory**: `numpy>=2.2.0`
- **Optional**: `matplotlib>=3.5.0` (plotting), `pytest>=9.0.0` (testing)
- **External**: `metaheuropt` (required by metaheuristic planners)

## Skills

- `.agents/skills/metaheuropt-path-planning/` — workflow for metaheuristic planners (GA, PSO, GWO).

## Style

- Full type hints on all functions/methods.
- Google-style docstrings.
- Prefer NumPy vectorized ops over loops.
