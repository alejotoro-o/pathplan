import gymnasium as gym
from gymnasium.spaces import Discrete
import numpy as np
from pathplan.core.map import GridMap

class PathPlanningEnv(gym.Env):
    """Gymnasium environment wrapping a GridMap for tabular RL path planning.

    The agent navigates a grid from start to goal using discrete moves.
    States are flattened cell indices ``row * W + col``.
    Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT.
    """

    def __init__(self, grid_map: GridMap, start, goal, threshold: float = 0.5):
        super().__init__()
        self.grid_map = grid_map
        self.H, self.W = grid_map.shape
        self.start = (int(start[0]), int(start[1]))
        self.goal = (int(goal[0]), int(goal[1]))
        self.threshold = threshold

        self.observation_space = Discrete(self.H * self.W)
        self.action_space = Discrete(4)
        self._actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def _get_state(self, row: int, col: int) -> int:
        return int(row * self.W + col)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._prev_state = self._get_state(*self.start)
        return self._prev_state, {}

    def step(self, action):
        row, col = divmod(self._prev_state, self.W)
        dr, dc = self._actions[action]
        nr, nc = row + dr, col + dc

        if not self.grid_map.is_valid_index(nr, nc) or \
           self.grid_map.is_occupied(nr, nc, threshold=self.threshold):
            next_state = self._get_state(row, col)
            reward = -10.0
            terminated = False
        elif (nr, nc) == self.goal:
            next_state = self._get_state(nr, nc)
            reward = 100.0
            terminated = True
        else:
            next_state = self._get_state(nr, nc)
            reward = -1.0
            terminated = False

        self._prev_state = next_state
        return next_state, reward, terminated, False, {}
