import torch
from utils.params import Params
import numpy as np


class _DuelingNetwork(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(_DuelingNetwork, self).__init__()
        self.feature = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
        )
        self.advantage = torch.nn.Linear(hidden_dim, out_dim)
        self.value = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()


def build_network_for(env, dueling=False):
    state_shape = env.observation_space.shape or (env.observation_space.n,)
    action_dim = env.action_space.n
    HIDDEN_LAYER_SIZE = Params.hidden_layer_size

    if dueling:
        net = _DuelingNetwork(
            np.prod(state_shape), HIDDEN_LAYER_SIZE, np.prod(action_dim)
        )
    else:
        net = torch.nn.Sequential(
            torch.nn.Linear(np.prod(state_shape), HIDDEN_LAYER_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_LAYER_SIZE, np.prod(action_dim)),
        )
    return net
