import numpy as np
from typing import Tuple, List, Optional, TYPE_CHECKING, Any

# This block is ONLY read by static type checkers (Mypy, Pyright, Pylance)
if TYPE_CHECKING:
    from matplotlib.axes import Axes

# Standard runtime imports
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class Visualizer:
    """Handles 2D rendering and analysis visualization for PathPlan objects."""
    
    def __init__(self, grid_map):
        if plt is None:
            raise ImportError(
                "Matplotlib is required to use the Visualizer. "
                "Please install it using: pip install pathplan[plot]"
            )
        self.grid_map = grid_map

    # Use a string literal "Axes" forward-reference to prevent runtime NameErrors
    def plot_map(self, ax: Optional["Axes"] = None) -> "Axes":
        """Renders the underlying grid map baseline layout."""
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))
            
        # Draw map using a clean inverted grayscale profile (0=white, 1=black)
        ax.imshow(self.grid_map.data, cmap="gray_r", origin="upper")
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    def plot_path(
        self, 
        path: List[Tuple[float, float]], 
        explored: Optional[Any] = None,
        title: str = "Path Planning Results"
    ) -> None:
        """Plots the final calculated trajectory path overlaying the map configuration."""
        _, ax = plt.subplots(figsize=(8, 8))
        self.plot_map(ax=ax)

        if explored:
            # Check if explored is a list of points or a list of edges
            if len(explored) > 0:
                if isinstance(explored[0][0], (int, float)):
                    # Points (for graph search)
                    explored_rows, explored_cols = zip(*explored)
                    ax.scatter(explored_cols, explored_rows, color="skyblue", s=15, alpha=0.4, label="Explored Nodes")
                else:
                    # Edges (for sampling search like RRT)
                    from matplotlib.collections import LineCollection
                    # Convert to (x, y) for matplotlib (which is col, row)
                    lines = [[(e[0][1], e[0][0]), (e[1][1], e[1][0])] for e in explored]
                    lc = LineCollection(lines, colors="skyblue", linewidths=1, alpha=0.4, label="Explored Edges")
                    ax.add_collection(lc)

        if path:
            # Matplotlib uses (x, y) which corresponds to (col, row)
            path_rows, path_cols = zip(*path)
            ax.plot(path_cols, path_rows, color="crimson", linewidth=3, label="Planned Path")
            ax.scatter(path_cols[0], path_rows[0], color="green", s=100, zorder=5, label="Start")
            ax.scatter(path_cols[-1], path_rows[-1], color="darkorange", s=100, zorder=5, label="Goal")

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.show()