import copy
import numpy as np
from replaybuffer import ReplayBuffer
import matplotlib.pyplot as plt
import torch
import random as r
import os
from os import path
from environments.maze_env import Maze
import random
import seaborn as sns
import socket
import wandb

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True  # Only this one is necessary for reproducibility
    # torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def to_numpy(tensor):
    if tensor is None:
        return None
    elif tensor.nelement() == 0:
        return np.array([])
    else:
        return tensor.cpu().detach().numpy()


def fill_buffer(buffer, num_transitions, env, noreset=False):
    if noreset:
        dont_take_reward = True
    else:
        dont_take_reward = False
        if env.name == 'maze':
            env.create_map()
            mode=1
        elif env.name == 'catcher':
            mode=-1
        else:
            pass

    end = num_transitions
    i = 0
    while i <= end:
        done = False
        state = env.observe()
        action = env.actions[r.randrange(env.num_actions)]
        reward = env.step(action, dont_take_reward=dont_take_reward)
        next_state = env.observe()
        if env.inTerminalState():
            env.reset(mode=1)
            done = True
        buffer.add(state, action, reward, next_state, done)
        i += 1
        if i >= end and not done:
            i -= 1

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)