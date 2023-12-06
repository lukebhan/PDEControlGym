import numpy as np

from .base_env_1d import PDEEnv1D

import gymnasium as gym
from gymnasium import spaces


class HyperbolicPDE1D(PDEEnv1D):
    def __init__(self, hyperbolicParams):
        super().__init__(hyperbolicParams)

    def step(self, control):
        Nx = self.parameters["nx"]
        dx = self.parameters["dx"]
        dt = self.parameters["dt"]
        self.time_index += 1
        # Explicit update of u according to finite difference derivation
        self.u[self.time_index][-1] = self.normalize(self.control_update(
            control, self.u[self.time_index][-2], self.parameters["dx"]), self.parameters["max_control_value"]
        )
        self.u[self.time_index][0 : Nx - 1] = self.u[self.time_index - 1][
            0 : Nx - 1
        ] + dt * (
            (
                self.u[self.time_index - 1][1:Nx]
                - self.u[self.time_index - 1][0 : Nx - 1]
            )
            / dx
            + (self.u[self.time_index - 1][0] * self.beta)[0 : Nx - 1]
        )
        terminate = self.terminate()
        truncate = self.truncate()
        return (
            self.sensing_update(
                self.u[self.time_index],
                self.parameters["dx"],
                self.parameters["sensing_noise_func"],
            ),
            self.reward.reward(self.u, self.time_index, terminate, truncate),
            terminate,
            truncate, 
            {},
        )

    def terminate(self):
        if self.time_index >= self.parameters["nt"] - 1:
            return True
        else:
            return False

    def truncate(self):
        if (
            self.parameters["limit_pde_state_size"]
            and np.linalg.norm(self.u[self.time_index], 2)/len(self.u[self.time_index]) >= self.parameters["max_state_value"]
        ):
            return True
        else:
            return False
         

    # Resets the system state
    def reset(self, seed=None, options=None):
        try:
            init_condition = self.parameters["reset_init_condition_func"](self.parameters["nx"])
            beta = self.parameters["reset_recirculation_func"](self.parameters["nx"], self.parameters["X"], self.parameters["reset_recirculation_param"])
        except:
            raise Exception(
                "Please pass both an initial condition and a recirculation function in the parameters dictionary. See documentation for more details"
                )
        self.u = np.zeros(
            (self.parameters["nt"], self.parameters["nx"]), dtype=np.float32
        )
        self.u[0] = init_condition
        self.time_index = 0
        self.beta = beta
        return (
            self.sensing_update(
                self.u[self.time_index],
                self.parameters["dx"],
                self.parameters["sensing_noise_func"],
            ),
            {},
        )
