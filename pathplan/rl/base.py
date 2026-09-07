from abc import abstractmethod
from typing import Tuple, List, Optional, Any

from pathplan.core.base_solver import BaseSolver
from pathplan.core.map import GridMap
from pathplan.rl.env import PathPlanningEnv
from pathplan.utils.geometry import edge_collision_free


class BaseRLPlanner(BaseSolver):
    """Abstract base for tabular RL path planners.

    Subclasses must implement :meth:`_create_agent` to return the
    RLForge agent instance.
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
        super().__init__(grid_map)
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.step_size = step_size
        self.discount = discount
        self.epsilon = epsilon
        self.threshold = occupancy_threshold

    @abstractmethod
    def _create_agent(self, num_states: int, num_actions: int):
        ...

    def _extract_path(
        self, agent, env: PathPlanningEnv, start, goal, max_steps: int = 500
    ) -> List[Tuple[int, int]]:
        agent.epsilon = 0.0
        state, _ = env.reset()
        path = [start]
        action = agent.start(state)

        for _ in range(max_steps):
            next_state, reward, terminated, _, _ = env.step(action)
            row, col = divmod(next_state, env.W)
            path.append((int(row), int(col)))
            if terminated:
                break
            action = agent.step(reward, next_state)

        return path

    def plan(
        self, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> Tuple[Optional[List[Tuple[int, int]]], Any]:
        if start == goal:
            return [start], []

        H, W = self.grid_map.shape
        env = PathPlanningEnv(self.grid_map, start, goal, threshold=self.threshold)
        agent = self._create_agent(H * W, 4)

        from rlforge.experiments import ExperimentRunner

        runner = ExperimentRunner(env, agent)
        runner.run_episodic(
            num_runs=1,
            num_episodes=self.num_episodes,
            max_steps_per_episode=self.max_steps,
        )

        path = self._extract_path(agent, env, start, goal)

        if path[-1] != goal:
            return None, path

        for i in range(len(path) - 1):
            if not edge_collision_free(
                path[i], path[i + 1], self.grid_map, threshold=self.threshold
            ):
                return None, path

        return path, path
