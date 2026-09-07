from typing import Tuple, List, Optional, Any

from pathplan.core.map import GridMap
from pathplan.rl.base import BaseRLPlanner

try:
    from rlforge.agents.tabular import SarsaAgent
except ImportError:
    raise ImportError(
        "rlforge library is required for SarsaPlanner. "
        "Please install it with: pip install rlforge"
    )


class SarsaPlanner(BaseRLPlanner):
    """Path planner using SARSA from the rlforge library.

    The agent learns a path by interacting with a grid-world environment
    and applying on-policy SARSA updates.
    """

    def __init__(
        self,
        grid_map: GridMap,
        num_episodes: int = 500,
        max_steps: int = 200,
        step_size: float = 0.1,
        discount: float = 0.95,
        epsilon: float = 0.1,
        occupancy_threshold: float = 0.5,
    ):
        super().__init__(
            grid_map,
            num_episodes=num_episodes,
            max_steps=max_steps,
            step_size=step_size,
            discount=discount,
            epsilon=epsilon,
            occupancy_threshold=occupancy_threshold,
        )

    def _create_agent(self, num_states: int, num_actions: int):
        return SarsaAgent(
            step_size=self.step_size,
            discount=self.discount,
            num_states=num_states,
            num_actions=num_actions,
            epsilon=self.epsilon,
        )

    def plan(
        self, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> Tuple[Optional[List[Tuple[int, int]]], Any]:
        return super().plan(start, goal)
