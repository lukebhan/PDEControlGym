import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

class HyperbolicPDEEnv(gym.Env):
    def __init__():
        self.observation_space = spaces.Dict(
                {
                    "u": spaces.Box(
