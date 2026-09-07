---
name: rlforge-path-planning
description: Workflow for implementing path planning agents using the rlforge reinforcement learning library. Use when developing new planners for PathPlan that learn navigation policies through tabular RL (Q-learning, SARSA).
---

# RL-Based Path Planning with RLForge (Tabular)

## Overview
This skill provides a workflow for implementing path planning algorithms using the `rlforge` reinforcement learning framework within `pathplan`. Unlike direct optimization, the agent learns a navigation policy through episodic interaction with a grid-world environment, then extracts a path via greedy rollout.

The documentation can be found in this [repository](https://github.com/alejotoro-o/rlforge) and [readthedocs](https://rlforge.readthedocs.io/).

## Imports
```python
import gymnasium as gym
from gymnasium.spaces import Discrete
import numpy as np
from rlforge.agents.tabular import QAgent, SarsaAgent, ExpectedSarsaAgent
from rlforge.experiments import ExperimentRunner
from pathplan.utils.geometry import edge_collision_free
```

## Dependencies
- `rlforge` (installs `numpy`, `gymnasium`, `matplotlib`, `tqdm`)
- PyTorch is **not** required for tabular agents

## Implementation Workflow

### 0. Edge Case: Start == Goal
If `start == goal`, return `[start]` immediately — no planning needed.

### 1. Custom GridWorld Environment
Create a Gymnasium environment that wraps `GridMap`. States are flattened cell indices `row * W + col`.

```python
class PathPlanningEnv(gym.Env):
    def __init__(self, grid_map, start, goal, max_steps=500):
        super().__init__()
        self.grid_map = grid_map
        self.H, self.W = grid_map.shape
        self.start = start
        self.goal = goal
        self.max_steps = max_steps
        self.steps = 0

        self.observation_space = Discrete(self.H * self.W)
        self.action_space = Discrete(4)  # UP, RIGHT, DOWN, LEFT
        self._actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def _get_state(self, row, col):
        return int(row * self.W + col)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return self._get_state(*self.start), {}

    def step(self, action):
        self.steps += 1
        row, col = divmod(self._prev_state, self.W)
        dr, dc = self._actions[action]
        nr, nc = row + dr, col + dc

        # Out of bounds or obstacle → stay, penalize
        if not self.grid_map.is_valid_index(nr, nc) or self.grid_map.is_occupied(nr, nc):
            reward = -10.0
            terminated = False
        # Reached goal
        elif (nr, nc) == self.goal:
            return self._get_state(nr, nc), 100.0, True, False, {}
        else:
            reward = -1.0
            terminated = False

        truncated = self.steps >= self.max_steps
        self._prev_state = self._get_state(nr, nc) if not terminated else self._prev_state
        return self._prev_state, reward, terminated, truncated, {}
```

### 2. Agent Setup
Choose a tabular agent and configure it for the grid dimensions. The epsilon parameter controls exploration during training.

```python
agent = QAgent(
    step_size=0.1,
    discount=0.95,
    num_states=H * W,
    num_actions=4,
    epsilon=0.1,
)
```

Available tabular agents:

| Import | Algorithm | Update rule |
|---|---|---|
| `QAgent` | Q-Learning | Off-policy, `r + γ max_a Q(s', a)` |
| `SarsaAgent` | SARSA | On-policy, `r + γ Q(s', a')` |
| `ExpectedSarsaAgent` | Expected SARSA | On-policy, `r + γ E[Q(s', ·)]` |
| `PlanningAgent` | Dyna-Q (base) | Adds model-based planning steps |

For Dyna-Q, add `planning=True, planning_steps=50` to the constructor.

### 3. Training
Wrap the environment and agent in `ExperimentRunner` and run episodic training.

```python
env = PathPlanningEnv(grid_map, start, goal)
runner = ExperimentRunner(env, agent)
results = runner.run_episodic(num_runs=1, num_episodes=500, max_steps_per_episode=200)
runner.plot_results()
runner.summary()
```

### 4. Path Extraction (Greedy Rollout)
After training, extract the path by following the learned policy greedily.

```python
def extract_path(agent, env, start, goal, max_steps=500):
    agent.epsilon = 0.0  # disable exploration for rollout
    agent.reset()
    state, _ = env.reset()
    path = [start]
    action = agent.start(state)

    for _ in range(max_steps):
        next_state, reward, terminated, _, _ = env.step(action)
        row, col = divmod(next_state, env.W)
        path.append((float(row), float(col)))
        if terminated:
            break
        action = agent.step(reward, next_state)

    return path
```

### 5. Collision Verification
Verify the extracted path with `edge_collision_free` for each consecutive pair.

### 6. PathPlan Integration
Wrap the full pipeline in a class extending `BaseSolver`.

```python
from pathplan.core.base_solver import BaseSolver
from pathplan.core.map import GridMap

class QLPlanner(BaseSolver):
    def __init__(self, grid_map: GridMap, num_episodes: int = 500,
                 step_size: float = 0.1, discount: float = 0.95,
                 epsilon: float = 0.1, max_steps: int = 200):
        super().__init__(grid_map)
        self.num_episodes = num_episodes
        self.step_size = step_size
        self.discount = discount
        self.epsilon = epsilon
        self.max_steps = max_steps

    def plan(self, start, goal):
        if start == goal:
            return [start], []

        H, W = self.grid_map.shape
        env = PathPlanningEnv(self.grid_map, start, goal,
                              max_steps=self.max_steps)
        agent = QAgent(step_size=self.step_size, discount=self.discount,
                       num_states=H * W, num_actions=4,
                       epsilon=self.epsilon)

        runner = ExperimentRunner(env, agent)
        runner.run_episodic(num_runs=1, num_episodes=self.num_episodes,
                            max_steps_per_episode=self.max_steps)

        path = extract_path(agent, env, start, goal)

        for i in range(len(path) - 1):
            if not edge_collision_free(path[i], path[i+1], self.grid_map):
                return None, path

        return path, path
```

## Agent Selection Guidance
- **`QAgent`**: Good default. Off-policy, learns the optimal path regardless of exploration.
- **`SarsaAgent`**: Better when learning during actual deployment (on-policy, considers exploration).
- **`ExpectedSarsaAgent`**: Lower variance updates than SARSA, slightly more stable.
- **`PlanningAgent`**: Use with `planning=True` for grid worlds with long corridors where model-based lookahead helps.
