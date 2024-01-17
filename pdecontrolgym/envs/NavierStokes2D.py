import numpy as np
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .util import central_difference_x, central_difference_y, laplace
from .base_env_2d import PDEEnv2D

import gymnasium as gym
from gymnasium import spaces


class Dirchilet:    # assume known
    def __init__(self, v):
        self.v = v

class Controllable: # the action that is optimizable
    def __init__(self, idx_dict):
        # idx_dict: a dictionary
        #   u: [0,1,2,...] index of control actions
        #   v: [...] index of control actions
        self.index = idx_dict

class Neumann:      # du/dx=0
    def __init__(self):
        pass


class NavierStokes2D(PDEEnv2D):
    def __init__(self, NSParams):
        super().__init__(NSParams)
        self.KINEMATIC_VISCOSITY = NSParams["viscosity"]
        self.DENSITY = NSParams["density"]
        self.N_PRESSURE_POISSON_ITERATIONS = NSParams["pressure_ite"]
        STABILITY_SAFETY_FACTOR = NSParams["stable_factor"]
        max_t = (0.5 * min(self.parameters["dx"], self.parameters["dy"])**2 / self.KINEMATIC_VISCOSITY)
        if self.parameters["dt"] > STABILITY_SAFETY_FACTOR * max_t:
            raise RuntimeError("Stability is not guarenteed")
        self.adapt = NSParams["adapt"]
        self.BoundaryControlInit()
    
    def BoundaryControlInit(self):
        self.control = self.parameters["control"]
        xx, yy = np.arange(0, self.parameters["nx"]), np.arange(0, self.parameters["ny"])
        self.pos_idx = {'lower': (0, xx), 'upper':(-1, xx), 'left': (yy, 0), 'right': (yy, -1)}
        self.pos_idx_neuman = {'lower': (1, xx), 'upper':(-2, xx), 'left': (yy, 1), 'right': (yy, -2)}

    def apply_boundary(self, u, v, action):
        action = self.adapt(action)
        for pos in ['lower', 'upper', 'left', 'right']:
            control_c = self.parameters["control"][pos]
            x_idx, y_idx = self.pos_idx[pos]
            if isinstance(control_c, Neumann):
                x_idx2, y_idx2 = self.pos_idx_neuman[pos]
                u[x_idx, y_idx] =  u[x_idx2, y_idx2] 
                v[x_idx, y_idx] =  u[x_idx2, y_idx2] 
            elif isinstance(control_c, Dirchilet):
                u[x_idx, y_idx] = control_c.v
                v[x_idx, y_idx] = control_c.v
            elif isinstance(control_c, Controllable):
                u[x_idx, y_idx] =  action[control_c.index['u']] 
                v[x_idx, y_idx] =  action[control_c.index['v']] 
        return u, v 
    

    def solve_pressure(self, u, v, p_prev):
        dx, dy, dt = self.parameters["dx"], self.parameters["dy"], self.parameters["dt"]
        dudx = central_difference_x(u, dx)
        dvdy = central_difference_y(v, dy)
        rhs = self.DENSITY / dt * (dudx + dvdy)
        for _ in range(self.N_PRESSURE_POISSON_ITERATIONS):
            p_next = p_prev.copy()
            p_next[1:-1, 1:-1] = 1/4 * (p_prev[1:-1, 0:-2] + p_prev[0:-2, 1:-1] + p_prev[1:-1, 2:  ] + p_prev[2:  , 1:-1]
                - dx * dy * rhs[1:-1, 1:-1]
            )
            # Neuman Condition for pressure
            p_next[:, -1] = p_next[:, -2]
            p_next[0,  :] = p_next[1,  :] 
            p_next[:,  0] = p_next[:,  1]
            p_next[-1, :] = p_next[-2, :]
            p_prev = p_next
        self.p = p_prev
        return p_prev

    def step(self, action):
        dx = self.parameters["dx"]
        dy = self.parameters["dy"]
        dt = self.parameters["dt"]
        u_prev, v_prev, p_prev = self.u, self.v, self.p
        dudx = central_difference_x(u_prev, dx)
        dudy = central_difference_y(u_prev, dy)
        dvdx = central_difference_x(v_prev, dx)
        dvdy = central_difference_y(v_prev, dy)
        laplace_u_prev = laplace(u_prev, dx, dy)
        laplace_v_prev = laplace(v_prev, dx, dy)
        # predictor step
        u_pred = u_prev + dt * (- u_prev * dudx - v_prev * dudy + self.KINEMATIC_VISCOSITY * laplace_u_prev)
        v_pred = v_prev + dt * (- u_prev * dvdx - v_prev * dvdy + self.KINEMATIC_VISCOSITY * laplace_v_prev)
        # apply boundary conditions
        u_pred, v_pred = self.apply_boundary(u_pred, v_pred, action)
        # solve for pressure
        pressure = self.solve_pressure(u_pred, v_pred, p_prev)
        dpdx, dpdy = central_difference_x(pressure, dx), central_difference_y(pressure, dy)
        u_next = u_pred - dt / self.DENSITY * dpdx
        v_next = v_pred - dt / self.DENSITY * dpdy
        u_next, v_next = self.apply_boundary(u_next, v_next, action)
        self.time_index += 1
        self.U[self.time_index, :, :, 0] = u_next
        self.U[self.time_index, :, :, 1] = v_next
        terminate = self.terminate()
        reward = - 1/2 * np.linalg.norm(self.U[self.time_index]-self.parameters["desire_U"][self.time_index])**2/21/21 - 0.1/2 * np.linalg.norm(action - 2.)**2
        self.u, self.v = u_next, v_next
        obs = self.U[self.time_index]
        truncated = False
        info = {}
        return obs, reward, terminate, truncated, info
    
    def terminate(self):
        if self.time_index >= self.parameters["nt"] - 1:
            return True
        else:
            return False

    # Resets the system state
    def reset(self, seed=None, options=None):
        self.U = np.zeros((self.parameters["nt"], self.parameters["nx"], self.parameters["ny"],  2))
        self.time_index = 0
        np.random.seed(seed)
        self.u = np.random.uniform(-5, 5) * np.ones_like(self.X) 
        self.v = np.random.uniform(-5, 5) * np.ones_like(self.X) 
        self.p = np.random.uniform(-5, 5) * np.ones_like(self.X)
        self.U[0,:,:,0] = self.u
        self.U[0,:,:,1] = self.v
        obs = self.U[self.time_index]
        return obs, {}