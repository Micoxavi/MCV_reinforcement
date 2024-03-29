"""
Script to try the working enviroment
"""

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gymnasium as gym

env = gym.make("PongNoFrameskip-v4", render_mode="human")
env.reset()

for i in range(3000):
    env.render()
    action = env.action_space.sample()
    next_state, reward, done, info, _ = env.step(action)
    print(f"nst: {next_state}, rw: {reward}, dn: {done}, info: {info}, else: {_}")

    if done:
        env.reset()
env.close()
