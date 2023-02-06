import gymnasium as gym
import torch
import torch.nn as nn
from torchinfo import summary

from src.reinforce import Reinforce

env = gym.make("CartPole-v1")
model = nn.Sequential(
    nn.Linear(4, 4),
    nn.LayerNorm(4),
    nn.GELU(),
    nn.Linear(4, 4),
    nn.LayerNorm(4),
    nn.GELU(),
    nn.Linear(4, 4),
    nn.LayerNorm(4),
    nn.GELU(),
    nn.Linear(4, 1),
)
summary(
    model,
    input_size=(4,),
    batch_dim=0,
    dtypes=[
        torch.float,
    ],
    device="cpu",
)

trainer = Reinforce(env, model, "cuda")
trainer.launch_training()
