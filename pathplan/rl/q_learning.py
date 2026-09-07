from typing import Tuple, List, Optional, Any

from pathplan.core.map import GridMap
from pathplan.rl.base import BaseRLPlanner

try:
    from rlforge.agents.tabular import QAgent
except ImportError:
    raise ImportError(
        "rlforge library is required for QLearningPlanner. "
        "Please install it with: pip install rlforge"
    )


class QLearningPlanner(BaseRLPlanner):
    """Path planner using Q-learning from the rlforge library.

    The agent learns an optimal path by interacting with a grid-world
    environment and applying off-policy Q-value updates.
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
        planning: bool = False,
        planning_steps: int = 0,
        exploration_bonus: float = 0.0,
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
        self.planning = planning
        self.planning_steps = planning_steps
        self.exploration_bonus = exploration_bonus

    def _create_agent(self, num_states: int, num_actions: int):
        return QAgent(
            step_size=self.step_size,
            discount=self.discount,
            num_states=num_states,
            num_actions=num_actions,
            epsilon=self.epsilon,
            planning=self.planning,
            planning_steps=self.planning_steps,
            exploration_bonus=self.exploration_bonus,
        )

    def plan(
        self, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> Tuple[Optional[List[Tuple[int, int]]], Any]:
        return super().plan(start, goal)
