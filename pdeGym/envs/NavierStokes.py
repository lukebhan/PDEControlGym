import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

from util import central_difference_x, central_difference_y, laplace, laplacian_operator
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib import cm

KINEMATIC_VISCOSITY = 0.1
DENSITY = 1.0
N_PRESSURE_POISSON_ITERATIONS = 50
STABILITY_SAFETY_FACTOR = 0.5


class NSPDEEnv(gym.Env):
    def __init__(self, T=1, dt=1e-3, X=1., dx=0.1, Y=1., dy=0.1):
        self.Nx, self.Ny = int(X / dx + 1), int(Y / dy + 1)
        self.dx, self.dy, self.dt = dx, dy, dt
        self.x = np.linspace(0, X, self.Nx)
        self.y = np.linspace(0, Y, self.Ny)
        self.X, self.Y = np.meshgrid(self.x, self.y[::-1])
        self.u_prev = np.zeros_like(self.X)
        self.v_prev = np.zeros_like(self.X)
        self.p_prev = np.zeros_like(self.X)
        max_t = (0.5 * min(dx,dy)**2 / KINEMATIC_VISCOSITY)
        self.lo = laplacian_operator(self.Nx, self.Ny, dx, dy)
        if dt > STABILITY_SAFETY_FACTOR * max_t:
            raise RuntimeError("Stability is not guarenteed")
    
    def step(self):
        u_prev, v_prev, p_prev = self.u_prev, self.v_prev, self.p_prev
        dx, dy, dt = self.dx, self.dy, self.dt
        dudx = central_difference_x(u_prev, dx)
        dudy = central_difference_y(u_prev, dy)
        dvdx = central_difference_x(v_prev, dx)
        dvdy = central_difference_y(v_prev, dy)
        laplace_u_prev = laplace(u_prev, dx)
        laplace_v_prev = laplace(v_prev, dy)
        # predictor step
        u_pred = u_prev + dt * (- u_prev * dudx - v_prev * dudy + KINEMATIC_VISCOSITY * laplace_u_prev)
        v_pred = v_prev + dt * (- u_prev * dvdx - v_prev * dvdy + KINEMATIC_VISCOSITY * laplace_v_prev)
        # apply boundary conditions
        u_pred, v_pred = self.apply_boundary(u_pred, v_pred)
        # solve for pressure
        pressure = self.solve_pressure(u_pred, v_pred, p_prev)
        dpdx, dpdy = central_difference_x(pressure, dx), central_difference_y(pressure, dy)
        u_next = u_pred - dt / DENSITY * dpdx
        v_next = v_pred - dt / DENSITY * dpdy
        u_next, v_next = self.apply_boundary(u_next, v_next)
        self.u_prev, self.v_prev = u_next, v_next
        return u_next, v_next

    def apply_boundary(self, u, v):
        u[0, :] = 0.0
        u[:, 0] = 0.0
        u[:, -1] = 0.0
        u[-1, :] = 0.
        v[0, :] = 1.0
        v[:, 0] = 0.0
        v[:, -1] = 0.0
        v[-1, :] = 0.
        return u, v 
    
    def solve_pressure(self, u, v, p_prev):
        dudx = central_difference_x(u, self.dx)
        dvdy = central_difference_y(v, self.dy)
        rhs = DENSITY / self.dt * (dudx + dvdy)
        for _ in range(N_PRESSURE_POISSON_ITERATIONS):
            p_next = np.zeros_like(p_prev)
            p_next[1:-1, 1:-1] = 1/4 * (
                +
                p_prev[1:-1, 0:-2]
                +
                p_prev[0:-2, 1:-1]
                +
                p_prev[1:-1, 2:  ]
                +
                p_prev[2:  , 1:-1]
                -
                self.dx * self.dy
                *
                rhs[1:-1, 1:-1]
            )
            p_next[:, -1] = p_next[:, -2]
            p_next[0,  :] = p_next[1,  :]
            p_next[:,  0] = p_next[:,  1]
            p_next[-1, :] = p_next[-2, :]
            p_prev = p_next
        self.p = p_prev
        return p_prev


if __name__ == "__main__":
    env = NSPDEEnv()
    for i in range(1000):
        env.step()
    plt.style.use("dark_background")
    plt.figure()
    plt.contourf(env.X, env.Y, env.p, cmap="coolwarm")
    plt.colorbar()
    plt.quiver(env.X, env.Y, env.u_prev, env.v_prev, color="black")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.show()
