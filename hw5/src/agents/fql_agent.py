from typing import Sequence

import infrastructure.pytorch_util as ptu
import numpy as np
import torch
from torch import nn


class FQLAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,
        make_bc_actor,
        make_bc_actor_optimizer,
        make_onestep_actor,
        make_onestep_actor_optimizer,
        make_critic,
        make_critic_optimizer,
        discount: float,
        target_update_rate: float,
        flow_steps: int,
        alpha: float,
    ):
        super().__init__()

        self.action_dim = action_dim

        self.bc_actor = make_bc_actor(observation_shape, action_dim)
        self.onestep_actor = make_onestep_actor(observation_shape, action_dim)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.bc_actor_optimizer = make_bc_actor_optimizer(self.bc_actor.parameters())
        self.onestep_actor_optimizer = make_onestep_actor_optimizer(
            self.onestep_actor.parameters()
        )
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.flow_steps = flow_steps
        self.alpha = alpha

    def get_action(self, observation: np.ndarray):
        """
        Used for evaluation.
        """
        observation_tensor = ptu.from_numpy(np.asarray(observation))[None]
        noise = torch.randn((1, self.action_dim), device=observation_tensor.device)
        action = self.onestep_actor(observation_tensor, noise)
        action = torch.clamp(action, -1, 1)
        return ptu.to_numpy(action)[0]

    @torch.compile
    def get_bc_action(self, observation: torch.Tensor, noise: torch.Tensor):
        """
        Used for training.
        """
        action = noise
        dt = 1.0 / self.flow_steps
        for step in range(self.flow_steps):
            times = torch.full_like(action[..., :1], step / self.flow_steps)
            action = action + dt * self.bc_actor(observation, action, times)
        action = torch.clamp(action, -1, 1)
        return action

    @torch.compile
    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update Q(s, a)
        """
        with torch.no_grad():
            noise = torch.randn_like(actions)
            next_actions = self.onestep_actor(next_observations, noise)
            next_actions = torch.clamp(next_actions, -1, 1)
            next_q = self.target_critic(next_observations, next_actions).mean(dim=0)
            target_q = rewards + self.discount * (1 - dones.float()) * next_q

        q = self.critic(observations, actions)
        loss = ((q - target_q.unsqueeze(0)) ** 2).mean()

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "q_loss": loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    @torch.compile
    def update_bc_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the BC actor
        """
        noise = torch.randn_like(actions)
        times = torch.rand_like(actions[..., :1])
        interpolated_actions = (1 - times) * noise + times * actions
        target_vectors = actions - noise
        pred_vectors = self.bc_actor(observations, interpolated_actions, times)
        loss = ((pred_vectors - target_vectors) ** 2).mean(dim=-1).mean()

        self.bc_actor_optimizer.zero_grad()
        loss.backward()
        self.bc_actor_optimizer.step()

        return {
            "loss": loss,
        }

    @torch.compile
    def update_onestep_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the one-step actor
        """
        noise = torch.randn_like(actions)
        onestep_actions = self.onestep_actor(observations, noise)

        with torch.no_grad():
            bc_actions = self.get_bc_action(observations, noise)

        distill_mses = ((onestep_actions - bc_actions) ** 2).mean(dim=-1)
        distill_loss = self.alpha * distill_mses.mean()

        critic_actions = torch.clamp(onestep_actions, -1, 1)
        q_loss = -self.critic(observations, critic_actions).mean(dim=0).mean()

        # Total loss.
        loss = distill_loss + q_loss

        # Additional metrics for logging.
        mse = ((onestep_actions - actions) ** 2).mean()

        self.onestep_actor_optimizer.zero_grad()
        loss.backward()
        self.onestep_actor_optimizer.step()

        return {
            "total_loss": loss,
            "distill_loss": distill_loss,
            "q_loss": q_loss,
            "mse": mse,
        }

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        metrics_q = self.update_q(
            observations, actions, rewards, next_observations, dones
        )
        metrics_bc_actor = self.update_bc_actor(observations, actions)
        metrics_onestep_actor = self.update_onestep_actor(observations, actions)
        metrics = {
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"bc_actor/{k}": v.item() for k, v in metrics_bc_actor.items()},
            **{
                f"onestep_actor/{k}": v.item() for k, v in metrics_onestep_actor.items()
            },
        }

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        for target_param, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1 - self.target_update_rate)
                + param.data * self.target_update_rate
            )
