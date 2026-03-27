from dataclasses import dataclass

import torch


@dataclass
class MCTSPlannerConfig:
    depth: int = 3
    branching: int = 8
    c_puct: float = 1.0
    rollout_steps: int = 1
    discount: float = 0.99


class MCTSPlanner:
    """A lightweight lookahead planner over TransitionRewardNet."""

    def __init__(self, transition_model, num_actions: int, config: MCTSPlannerConfig):
        self.transition_model = transition_model
        self.num_actions = num_actions
        self.config = config

    @torch.no_grad()
    def plan(self, world_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            world_state: [bs, state_dim]
        Returns:
            action_ids: [bs]
        """
        bs = world_state.shape[0]
        device = world_state.device
        action_space = torch.arange(self.num_actions, device=device, dtype=torch.long)

        best_action = torch.zeros(bs, dtype=torch.long, device=device)
        best_value = torch.full((bs,), -1e9, device=device)
        for action in action_space:
            act = torch.full((bs,), int(action.item()), dtype=torch.long, device=device)
            v = self._rollout(world_state, act, self.config.depth)
            better = v > best_value
            best_value = torch.where(better, v, best_value)
            best_action = torch.where(better, act, best_action)
        return best_action

    def _rollout(self, world_state: torch.Tensor, action_ids: torch.Tensor, depth: int) -> torch.Tensor:
        next_state, reward = self.transition_model(world_state, action_ids)
        if depth <= 1:
            return reward
        candidates = torch.randperm(self.num_actions, device=world_state.device)[: min(self.config.branching, self.num_actions)]
        best_future = torch.full_like(reward, -1e9)
        for a in candidates:
            act = torch.full_like(action_ids, int(a.item()))
            future = self._rollout(next_state, act, depth - 1)
            best_future = torch.maximum(best_future, future)
        return reward + self.config.discount * best_future
