from typing import List, Optional, Tuple, TYPE_CHECKING

from pathplan.core.map import GridMap
from pathplan.multiagent.utils.utils import detect_all_collisions

if TYPE_CHECKING:
    from matplotlib.axes import Axes

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class MultiAgentVisualizer:
    """2D rendering for multi-agent path planning results.

    Mirrors the single-agent :class:`pathplan.utils.Visualizer` API
    but handles multiple agent paths on a single figure.

    Parameters
    ----------
    grid_map:
        The shared occupancy grid.
    """

    COLORS: List[str] = [
        "crimson",
        "dodgerblue",
        "forestgreen",
        "darkorange",
        "mediumpurple",
        "saddlebrown",
        "deeppink",
        "teal",
    ]

    def __init__(self, grid_map: GridMap) -> None:
        if plt is None:
            raise ImportError(
                "Matplotlib is required to use the MultiAgentVisualizer. "
                "Please install it using: pip install pathplan[plot]"
            )
        self.grid_map = grid_map

    def plot_map(self, ax: Optional["Axes"] = None) -> "Axes":
        """Render the grid map baseline layout."""
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self.grid_map.data, cmap="gray_r", origin="upper")
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    def plot_paths(
        self,
        paths: List[Optional[List[Tuple[int, int]]]],
        title: str = "Multi-Agent Paths",
    ) -> None:
        """Plot all agent paths in different colors on one figure.

        Parameters
        ----------
        paths:
            One path per agent.  ``None`` entries are skipped with a
            console warning.
        title:
            Figure title.
        """
        _, ax = plt.subplots(figsize=(8, 8))
        self.plot_map(ax=ax)

        for i, path in enumerate(paths):
            color = self.COLORS[i % len(self.COLORS)]
            if path is None:
                print(f"  Agent {i}: no path found — skipped")
                continue
            if len(path) == 0:
                continue

            rows, cols = zip(*path)
            ax.plot(
                cols, rows,
                color=color, linewidth=2, label=f"Agent {i}",
            )
            ax.scatter(
                cols[0], rows[0],
                color=color, s=80, zorder=5, marker="o",
            )
            ax.scatter(
                cols[-1], rows[-1],
                color=color, s=80, zorder=5, marker="x",
            )

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        plt.tight_layout()
        plt.show()

    def plot_paths_with_collisions(
        self,
        paths: List[Optional[List[Tuple[int, int]]]],
        title: str = "Multi-Agent Paths (with collisions)",
    ) -> None:
        """Like :meth:`plot_paths` but highlights collision points.

        Collision locations are marked with large red circles.
        """
        _, ax = plt.subplots(figsize=(8, 8))
        self.plot_map(ax=ax)

        valid_paths = [p for p in paths if p is not None]

        for i, path in enumerate(paths):
            color = self.COLORS[i % len(self.COLORS)]
            if path is None:
                print(f"  Agent {i}: no path found — skipped")
                continue
            if len(path) == 0:
                continue

            rows, cols = zip(*path)
            ax.plot(
                cols, rows,
                color=color, linewidth=2, label=f"Agent {i}",
            )
            ax.scatter(
                cols[0], rows[0],
                color=color, s=80, zorder=5, marker="o",
            )
            ax.scatter(
                cols[-1], rows[-1],
                color=color, s=80, zorder=5, marker="x",
            )

        # Highlight collisions
        if len(valid_paths) >= 2:
            collisions = detect_all_collisions(valid_paths)
            for col in collisions:
                for loc in col["loc"]:
                    ax.scatter(
                        loc[1], loc[0],
                        color="red", s=200, zorder=6,
                        edgecolors="black", linewidths=1.5,
                        label="Collision" if col == collisions[0] else "",
                    )

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        plt.tight_layout()
        plt.show()
