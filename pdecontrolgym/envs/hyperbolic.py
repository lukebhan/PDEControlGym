import numpy as np

from .base_env_1d import PDEEnv1D

import gymnasium as gym
from gymnasium import spaces


class HyperbolicPDE1D(PDEEnv1D):
    def __init__(self, hyperbolicParams):
        super().__init__(hyperbolicParams)
	  # Observation space changes depending on sensing
        match self.parameters["sensing_loc"]:
            case "full":
                self.observation_space = spaces.Box(
                    np.full(self.parameters["nx"], -self.parameters["max_state_value"], dtype="float32"),
                    np.full(self.parameters["nx"], self.parameters["max_state_value"], dtype="float32"),
                )
            case "collocated" | "opposite":
                self.observation_space = spaces.Box(
                    np.full(1, -self.parameters["max_state_value"], dtype="float32"),
                    np.full(1, self.parameters["max_state_value"], dtype="float32"),
                )
            case _:
                raise Exception(
                    "Invalid sensing_loc parameter. Please use 'full', 'collocated', or 'opposite'. See documentation for details."
                )

        # Setup configurations for control and sensing. Messy, but done once, explicitly before runtime to setup return and control functions
        # There is a trick here where noise is a function call itself. Important that noise is a single argument function that returns a single argument
        match self.parameters["control_type"]:
            case "Neumann":
                self.control_update = lambda control, state, dx: control * dx + state
                match self.parameters["sensing_loc"]:
                    # Neumann control u_x(1), full state measurement
                    case "full":
                        self.sensing_update = lambda state, dx, noise: noise(state)
                    # Neumann control u_x(1), Dirchilet sensing u(1)
                    case "collocated":
                        self.sensing_update = lambda state, dx, noise: noise(state[-1])
                    case "opposite":
                        match "sensing_type":
                            # Neumann control u_x(1), Neumann sensing u_x(0)
                            case "Neumann":
                                self.sensing_update = lambda state, dx, noise: noise(
                                    (state[1] - state[0]) / dx
                                )
                            # Neumann control u_x(1), Dirchilet sensing u(0)
                            case "Dirchilet":
                                self.sensing_update = lambda state, dx, noise: noise(state[0])
                            case _:
                                raise Exception(
                                    "Invalid sensing_type parameter. Please use 'Neumann' or 'Dirchilet'. See documentation for details."
                                )
                    case _:
                        raise Exception(
                            "Invalid sensing_loc parameter. Please use 'full', 'collocated', or 'opposite'. See documentation for details."
                        )
            case "Dirchilet":
                self.control_update = lambda control, state, dt: control
                match self.parameters["sensing_loc"]:
                    # Dichilet control u(1), full state measurement
                    case "full":
                        self.sensing_update = lambda state, dx, noise: noise(state)
                    # Dichilet control u(1), Neumann sensing u_x(1)
                    case "collocated":
                        self.sensing_update = lambda state, dx, noise: noise(
                            (state[-1] - state[-2]) / dx
                        )
                    case "opposite":
                        match "sensing_type":
                            # Dichilet control u(1), Neumann sensing u_x(0)
                            case "Neumann":
                                self.sensing_update = lambda state, dx, noise: noise(
                                    (state[1] - state[0]) / dx
                                )
                            # Dirchilet control u(1), Dirchilet sensing u(0)
                            case "Dirchilet":
                                self.sensing_update = lambda state, dx, noise: noise(
                                    state[0]
                                )
                            case _:
                                raise Exception(
                                    "Invalid sensing_type parameter. Please use 'Neumann' or 'Dirchilet'. See documentation for details."
                                )
            case _:
                raise Exception(
                    "Invalid control_type parameter. Please use 'Neumann' or 'Dirchilet'. See documentation for details."
                )

    def step(self, control):
        Nx = self.parameters["nx"]
        dx = self.parameters["dx"]
        dt = self.parameters["dt"]
        sample_rate = int(round(self.parameters["control_sample_rate"]/dt))
        i = 0
        # Actions are applied at a slower rate then the PDE is simulated at
        while i < sample_rate and self.time_index < self.parameters["nt"]-1:
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
            i += 1
        terminate = self.terminate()
        truncate = self.truncate()
        return (
            self.sensing_update(
                self.u[self.time_index],
                self.parameters["dx"],
                self.parameters["sensing_noise_func"],
            ),
            self.reward.shapedReward(self.u, self.time_index, terminate, truncate, self.u[self.time_index][-1]),
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
        #if (
        #    self.parameters["limit_pde_state_size"]
        #    and np.linalg.norm(self.u[self.time_index], 2)  >= self.parameters["max_state_value"]
        #):
        #    return True
        #else:
        return False
         

    # Resets the system state
    def reset(self, seed=None, options=None):
        try:
            init_condition = self.parameters["reset_init_condition_func"](self.parameters["nx"])
            beta = self.parameters["reset_recirculation_func"](self.parameters["nx"], self.parameters["X"])
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
