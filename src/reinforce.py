from itertools import accumulate

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli
from torch.nn.utils.clip_grad import clip_grad_norm_


class Reinforce:
    def __init__(
        self,
        env: gym.Env,
        model: nn.Module,
        device: str,
    ):
        self.env = env
        self.model = model
        self.device = device

        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3)
        self.gamma = 0.99

        self.episodes_history = []

    @staticmethod
    def select_tile(logits: torch.Tensor) -> tuple[int, torch.Tensor]:
        distribution = Bernoulli(logits=logits)
        action = distribution.sample()
        log_action = distribution.log_prob(action)

        return action.cpu().long().item(), log_action

    def rollout(self):
        """Do a simple episode rollout using the current model's policy.

        It saves the history so that the gradient can be later computed.
        """
        env, model, device = self.env, self.model, self.device

        state, _ = env.reset()
        done = False

        rewards, log_actions = [], []
        while not done:
            state = torch.FloatTensor(state).to(device).unsqueeze(0)
            logits = model(state)
            action, log_act = Reinforce.select_tile(logits)
            log_actions.append(log_act)

            state, reward, terminated, truncated, *_ = env.step(action)
            done = terminated or truncated
            rewards.append(reward)

        rewards = torch.tensor(rewards, device=device)  # Shape of [ep_len,].
        log_actions = torch.concat(log_actions, dim=0)  # Shape of [ep_len, 1].

        # Compute the cumulated returns.
        rewards = torch.flip(rewards, dims=(0,))
        returns = list(accumulate(rewards, lambda R, r: r + self.gamma * R))
        returns = torch.tensor(returns, device=device)
        returns = torch.flip(returns, dims=(0,))

        self.episodes_history.append((log_actions, returns))

    def compute_metrics(self) -> dict[str, torch.Tensor]:
        loss = torch.tensor(0.0, device=self.device)
        history_returns = torch.zeros(len(self.episodes_history), device=self.device)

        mean_returns = sum(returns.sum() for _, returns in self.episodes_history)
        mean_returns = mean_returns / sum(
            len(returns) for _, returns in self.episodes_history
        )

        for ep_id, (log_actions, returns) in enumerate(self.episodes_history):
            history_returns[ep_id] = returns[0]
            # returns = returns - mean_returns
            loss += -(log_actions * returns.unsqueeze(1)).mean()

        metrics = {
            "loss": loss,
            "return": history_returns.mean(),
        }
        return metrics

    def launch_training(self):
        optim = self.optimizer
        self.model.to(self.device)

        n_batches = 2000
        n_rollouts = 200

        for _ in range(n_batches):
            self.episodes_history = []

            for _ in range(n_rollouts):
                self.rollout()

            metrics = self.compute_metrics()
            optim.zero_grad()
            metrics["loss"].backward()
            clip_grad_norm_(self.model.parameters(), 1)
            optim.step()

            for metric_name in ["loss", "return"]:
                value = metrics[metric_name].cpu().item()
                print(f"{metric_name}: {value:.3f}", end="\t")
            print("")
