import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Callable, Optional, Union

from pde_control_gym.src.environments2d.base_env_2d import PDEEnv2D


def central_difference(f, coordinate, step=0.01):
    diff = np.zeros_like(f)
    if coordinate == "x":
        diff[1:-1, 1:-1] = (f[1:-1, 2:] - f[1:-1, 0:-2]) / (2 * step)
    elif coordinate == "y":
        diff[1:-1, 1:-1] = (f[2:, 1:-1] - f[0:-2, 1:-1]) / (2 * step)
    return diff

def laplace(f, dx=0.01, dy=0.01):
    diff = np.zeros_like(f)
    diff[1:-1, 1:-1] = (
        f[1:-1, 0:-2] + f[0:-2, 1:-1] - 4 * f[1:-1, 1:-1] + f[1:-1, 2:] + f[2:, 1:-1]
    ) / (dx * dy)
    return diff

class NavierStokes2D(PDEEnv2D):
    """
    NavierStokes equations 2D
    This class implements the 2D NavierStokes PDE and inhertis from the class :class:`PDEEnv2D`. Thus, for a full list of of arguments, first see the class :class:`PDEEnv2D` in conjunction with the arguments presented here

    :param reset_init_condition_func: Takes in a function used during the reset method for setting the initial PDE condition :math:`U(x, y, 0)=[u(x,y,0), v(x,y,0)]`.
    :param boundary_condition: dictionary recording that the top/bottom/left/right is at what condition: Neumann/Dirchilet/Controllable
    :param U_ref: reference trajectory of PDEs
    :param action_ref: reference actions of PDEs
    :param viscosity: kinematic viscosity value for the NavierStokes PDE
    :param dentisty: density value for pressure field in the NavierStokes PDE
    :param maximum_pressure_iteration:  the maximum iterations to solve for the pressure field  
    :param stable_factor: the stability factor for the stability of NavierStokes
    """
    def __init__(self, reset_init_condition_func: Callable[[int], np.ndarray],
                 boundary_condition: dict,
                 U_ref: np.ndarray, 
                 action_ref: np.ndarray,
                 viscosity: float = 0.1,
                 density: float = 1.0, 
                 maximum_pressure_iteration: float = 2000,
                 stable_factor: float = 0.5,
                 **kwargs
                ):
        super().__init__(**kwargs)
        self.reset_init_condition_func = reset_init_condition_func 
        self.KINEMATIC_VISCOSITY = viscosity
        self.DENSITY = density
        self.N_PRESSURE_POISSON_ITERATIONS = maximum_pressure_iteration
        STABILITY_SAFETY_FACTOR = stable_factor
        self.U_ref = U_ref
        self.action_ref = action_ref
        max_t = (0.5 * min(self.dx, self.dy)**2 / self.KINEMATIC_VISCOSITY)
        if self.dt > STABILITY_SAFETY_FACTOR * max_t:
            raise RuntimeError("Stability is not guarenteed")
        self.BoundaryControlInit(boundary_condition)
    
    def BoundaryControlInit(self, boundary_condition: dict):
        # Setup configurations of boundary conditions
        self.boundary_condition = boundary_condition
        xx, yy = np.arange(0, self.nx), np.arange(0, self.ny)
        self.pos_idx = {'lower': (0, xx), 'upper':(-1, xx), 'left': (yy, 0), 'right': (yy, -1)}
        self.pos_idx_neuman = {'lower': (1, xx), 'upper':(-2, xx), 'left': (yy, 1), 'right': (yy, -2)}

    def apply_boundary(self, u: np.ndarray, v: np.ndarray, action: Union[float, np.ndarray]):
        """
        Each time solving the PDE, we apply the boundary conditions to u and v
        
        :param u: :math:`u(x,y)`
        :param v: :math:`v(x,y)`
        :param action: action performed by reinforcement learning
        """
        for pos in ['lower', 'upper', 'left', 'right']:
            for i in range(2): 
                condition = self.boundary_condition[pos][i]
                xidx, yidx = self.pos_idx[pos]
                match condition:
                    case "Neumann":
                        xidx2, yidx2 = self.pos_idx_neuman[pos]
                        if i == 0: u[xidx, yidx] = u[xidx2, yidx2]
                        else: v[xidx, yidx] = v[xidx2, yidx2]
                    case "Dirchilet":
                        if i == 0: u[xidx, yidx] = 0
                        else: v[xidx, yidx] = 0
                    case "Controllable":
                        if i == 0: u[xidx, yidx] = action
                        else: v[xidx, yidx] = action
        return u, v 
    

    def solve_pressure(self, u: np.ndarray, v: np.ndarray, p_prev: np.ndarray):
        """
        Solving pressure

        Usinf an iterative approach to solve pressure
        """
        dx, dy, dt = self.dx, self.dy, self.dt
        dudx = central_difference(u,"x", dx)
        dvdy = central_difference(v,"y", dy)
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

    def step(self, action:Union[float, np.ndarray]):
        """
        step

        Moves the PDE with control action forward ``dt`` steps.

        :param action: the control action to apply to the PDE at the boundary.
        """
        dx = self.dx
        dy = self.dy
        dt = self.dt
        u_prev, v_prev, p_prev = self.u, self.v, self.p
        dudx = central_difference(u_prev, "x", dx)
        dudy = central_difference(u_prev, "y", dy)
        dvdx = central_difference(v_prev, "x", dx)
        dvdy = central_difference(v_prev, "y", dy)
        laplace_u_prev = laplace(u_prev, dx, dy)
        laplace_v_prev = laplace(v_prev, dx, dy)
        # predictor step
        u_pred = u_prev + dt * (- u_prev * dudx - v_prev * dudy + self.KINEMATIC_VISCOSITY * laplace_u_prev)
        v_pred = v_prev + dt * (- u_prev * dvdx - v_prev * dvdy + self.KINEMATIC_VISCOSITY * laplace_v_prev)
        # apply boundary conditions
        u_pred, v_pred = self.apply_boundary(u_pred, v_pred, action)
        # solve for pressure
        pressure = self.solve_pressure(u_pred, v_pred, p_prev)
        dpdx, dpdy = central_difference(pressure, "x", dx), central_difference(pressure, "y", dy)
        u_next = u_pred - dt / self.DENSITY * dpdx
        v_next = v_pred - dt / self.DENSITY * dpdy
        u_next, v_next = self.apply_boundary(u_next, v_next, action)
        self.time_index += 1
        self.U[self.time_index, :, :, 0] = u_next
        self.U[self.time_index, :, :, 1] = v_next
        terminate = self.terminate()
        reward = self.reward_class.reward(self.U, self.time_index, self.U_ref, action, self.action_ref)
        #- 1/2 * np.linalg.norm(self.U[self.time_index]-self.desired_U[self.time_index])**2/21/21 - 0.1/2 * np.linalg.norm(action - 2.)**2
        self.u, self.v = u_next, v_next
        obs = self.U[self.time_index]
        truncated = False
        info = {}
        return obs, reward, terminate, truncated, info
    
    def terminate(self):
        """
        terminate

        Determines whether the episode should end if the ``T`` timesteps are reached
        """
        if self.time_index >= self.nt - 1:
            return True
        else:
            return False

    # Resets the system state
    def reset(self, seed: Optional[int]=None, options: Optional[dict]=None):
        """
        reset 

        :param seed: Allows a seed for initialization of the envioronment to be set for RL algorithms.
        :param options: Allows a set of options for the initialization of the environment to be set for RL algorithms.

        Resets the PDE at the start of each environment according to the parameters given during the PDE environment intialization
        """
        try:
            init_u, init_v, init_p = self.reset_init_condition_func(self.X)
        except:
            raise Exception(
                "Please pass both an initial condition and a recirculation function in the parameters dictionary. See documentation for more details"
                )
        self.U = np.zeros((self.nt, self.nx, self.ny,  2))
        self.time_index = 0
        self.u = init_u
        self.v = init_v # np.random.uniform(-5, 5) * np.ones_like(self.X) 
        self.p = init_p
        self.U[0,:,:,0] = init_u
        self.U[0,:,:,1] = init_v
        obs = self.U[self.time_index]
        return obs, {}
