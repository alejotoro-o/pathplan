import heapq
import math
from typing import Any, Dict, List, Optional, Set, Tuple

from pathplan.core.map import GridMap
from pathplan.multiagent.core.base import Agents, BaseMAPFSolver, Path
from pathplan.multiagent.utils.utils import (
    ReservationTable,
    build_constraint_table,
    detect_all_collisions,
    is_constrained,
    space_time_a_star,
)


def _path_cost(path: List[Tuple[int, int]]) -> float:
    cost = 0.0
    for i in range(1, len(path)):
        dr = abs(path[i][0] - path[i - 1][0])
        dc = abs(path[i][1] - path[i - 1][1])
        cost += math.sqrt(2) if (dr + dc == 2) else 1.0
    return cost


def _build_priority_constraints(
    agent_id: int,
    higher_agents: List[int],
    all_paths: List[Optional[Path]],
) -> List[Dict]:
    """Build constraints so that *agent_id* avoids all higher-priority agents."""
    constraints: List[Dict] = []
    for j in higher_agents:
        pj = all_paths[j]
        if pj is None:
            continue
        for t in range(len(pj)):
            loc = pj[t]
            constraints.append({
                "agent": agent_id,
                "loc": [loc],
                "timestep": t,
                "type": "vertex",
            })
            if t > 0:
                prev = pj[t - 1]
                constraints.append({
                    "agent": agent_id,
                    "loc": [prev, loc],
                    "timestep": t,
                    "type": "edge",
                })
    return constraints


class _PBSNode:
    """Node in the PBS priority tree."""

    __slots__ = ("priorities", "paths", "cost")

    def __init__(
        self,
        priorities: Dict[int, List[int]],
        paths: List[Optional[Path]],
        cost: float,
    ) -> None:
        self.priorities = priorities  # {rank: [agent_ids]}
        self.paths = paths
        self.cost = cost

    def __lt__(self, other: "_PBSNode") -> bool:
        return self.cost < other.cost


class PBSPlanner(BaseMAPFSolver):
    """Priority-Based Search — tree search over priority orderings.

    Unlike :class:`PrioritizedPlanner` which uses a single fixed ordering,
    PBS searches over partial priority orderings to find a collision-free
    solution.  More complete than PrioritizedPlanning but still suboptimal.

    Parameters
    ----------
    grid_map:
        The shared occupancy grid.
    max_timesteps:
        Planning horizon for each low-level search.
    """

    def __init__(self, grid_map: GridMap, max_timesteps: int = 500) -> None:
        super().__init__(grid_map, max_timesteps)

    def _plan_with_priorities(
        self,
        agents: Agents,
        ordered_agents: List[int],
    ) -> List[Optional[Path]]:
        """Plan agents sequentially according to *ordered_agents*."""
        rt = ReservationTable()
        paths: List[Optional[Path]] = [None] * len(agents)
        ct_all: List[Dict] = []  # accumulated constraints from all higher agents

        for agent_id in ordered_agents:
            start, goal = agents[agent_id]
            # Build constraint table for this agent from all higher-priority agents
            ct = build_constraint_table(ct_all, agent_id)
            inf_cons: Dict[Tuple[int, int], int] = {}
            for t, clist in ct.items():
                for c in clist:
                    if c["type"] == "inf":
                        pos = c["loc"][0]
                        prev = inf_cons.get(pos)
                        if prev is None or t < prev:
                            inf_cons[pos] = t

            path = space_time_a_star(
                self.grid_map, start, goal, rt,
                max_timesteps=self.max_timesteps,
                constraint_table=ct,
                inf_constraints=inf_cons if inf_cons else None,
            )
            if path is None:
                return paths
            paths[agent_id] = path
            # Reserve path and add constraints for lower-priority agents
            rt.reserve_path(path)
            for t in range(len(path)):
                loc = path[t]
                ct_all.append({
                    "agent": -1,  # placeholder — will be filtered by agent_id
                    "loc": [loc],
                    "timestep": t,
                    "type": "vertex",
                })
                if t > 0:
                    ct_all.append({
                        "agent": -1,
                        "loc": [path[t - 1], loc],
                        "timestep": t,
                        "type": "edge",
                    })
        return paths

    def plan(self, agents: Agents) -> Tuple[List[Optional[Path]], Any]:
        n = len(agents)
        if n == 0:
            return [], {}

        if n == 1:
            paths = self._plan_with_priorities(agents, [0])
            return paths, {"nodes_expanded": 0, "nodes_generated": 0}

        # Root: empty priority ordering — plan agent 0 first
        root_paths = [None] * n
        rt = ReservationTable()
        path0 = space_time_a_star(
            self.grid_map, agents[0][0], agents[0][1], rt,
            max_timesteps=self.max_timesteps,
        )
        if path0 is None:
            return [None] * n, {"nodes_expanded": 0, "nodes_generated": 0}
        root_paths[0] = path0
        rt.reserve_path(path0)

        # Try to plan remaining agents; collect unprioritized ones
        unprioritized: List[int] = []
        for i in range(1, n):
            pi = space_time_a_star(
                self.grid_map, agents[i][0], agents[i][1], rt,
                max_timesteps=self.max_timesteps,
            )
            if pi is not None:
                root_paths[i] = pi
                rt.reserve_path(pi)
            else:
                unprioritized.append(i)

        # If all planned, check for collisions
        collisions = detect_all_collisions([p for p in root_paths if p is not None])
        root_cost = sum(_path_cost(p) for p in root_paths if p is not None)

        if not collisions and not unprioritized:
            return root_paths, {"nodes_expanded": 0, "nodes_generated": 0}

        # Build root priority ordering
        ordered = [i for i in range(n) if root_paths[i] is not None]
        root_prio: Dict[int, List[int]] = {}
        for rank, aid in enumerate(ordered):
            root_prio[rank] = [aid]

        root = _PBSNode(root_prio, root_paths, root_cost)

        open_list: list = [root]
        nodes_expanded = 0
        nodes_generated = 1

        while open_list:
            node = heapq.heappop(open_list)
            nodes_expanded += 1

            # Check if solution is collision-free
            valid_paths = [p for p in node.paths if p is not None]
            if valid_paths:
                collisions = detect_all_collisions(valid_paths)
            else:
                collisions = []

            if not collisions:
                # Check all agents are planned
                if all(p is not None for p in node.paths):
                    return node.paths, {
                        "nodes_expanded": nodes_expanded,
                        "nodes_generated": nodes_generated,
                    }

            # Find an agent that conflicts or is unplanned
            # Try to add the next unplanned agent or resolve a conflict
            target_agent = None
            for i in range(n):
                if node.paths[i] is None:
                    target_agent = i
                    break

            if target_agent is None and collisions:
                # Pick agents from first collision
                first = collisions[0]
                target_agent = first["a1"]  # try replanning a1

            if target_agent is None:
                continue

            # Find which agents are higher priority than target_agent
            higher: List[int] = []
            for rank, agents_at_rank in node.priorities.items():
                for aid in agents_at_rank:
                    if aid != target_agent:
                        higher.append(aid)

            # Replan target_agent with higher-priority constraints
            start, goal = agents[target_agent]
            ct = _build_priority_constraints(target_agent, higher, node.paths)
            ct_table = build_constraint_table(ct, target_agent)
            inf_cons: Dict[Tuple[int, int], int] = {}
            for t, clist in ct_table.items():
                for c in clist:
                    if c["type"] == "inf":
                        pos = c["loc"][0]
                        prev = inf_cons.get(pos)
                        if prev is None or t < prev:
                            inf_cons[pos] = t

            new_path = space_time_a_star(
                self.grid_map, start, goal, ReservationTable(),
                max_timesteps=self.max_timesteps,
                constraint_table=ct_table,
                inf_constraints=inf_cons if inf_cons else None,
            )

            if new_path is not None:
                new_paths = list(node.paths)
                new_paths[target_agent] = new_path
                new_cost = sum(_path_cost(p) for p in new_paths if p is not None)

                # Update priority ordering
                new_prio = dict(node.priorities)
                assigned_ranks = [r for r in sorted(new_prio.keys())]
                next_rank = max(assigned_ranks) + 1 if assigned_ranks else 0
                new_prio[next_rank] = [target_agent]

                child = _PBSNode(new_prio, new_paths, new_cost)
                heapq.heappush(open_list, child)
                nodes_generated += 1

        return [None] * n, {
            "nodes_expanded": nodes_expanded,
            "nodes_generated": nodes_generated,
        }
