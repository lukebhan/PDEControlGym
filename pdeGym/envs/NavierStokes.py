import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from base_env import PDECEnv

from util import central_difference_x, central_difference_y, laplace
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.sparse import lil_matrix
import os

KINEMATIC_VISCOSITY = 0.1
DENSITY = 1.0
N_PRESSURE_POISSON_ITERATIONS = 1000
STABILITY_SAFETY_FACTOR = 0.5


class NSPDEEnv(PDECEnv):
    def __init__(self, T=0.2, dt=1e-3, X=1., dx=0.1, Y=1., dy=0.1, plot=False):
        # PDE Settings (given as a dictionary for the first argument)
            # u - first dimension in velocity field, shape: (Nx, Ny)
            # v - second dimension in velocity field, shape: (Nx, Ny)
            # p - pressure field: shape (Nx, Ny)
        super(NSPDEEnv, self).__init__()
        self.Nx, self.Ny = int(X / dx + 1), int(Y / dy + 1)
        self.dx, self.dy, self.dt = dx, dy, dt
        self.x = np.linspace(0, X, self.Nx)
        self.y = np.linspace(0, Y, self.Ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        max_t = (0.5 * min(self.dx, self.dy)**2 / KINEMATIC_VISCOSITY)
        if dt > STABILITY_SAFETY_FACTOR * max_t:
            raise RuntimeError("Stability is not guarenteed")
        self.t, self.T = 0., T
        self.reset()
        # TODO: 
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(8*self.Nx, ), dtype=np.float32)
        self.observation_space = spaces.Box(low=-100., high=100., shape=(self.Nx, self.Nx, 2), dtype=np.float32)
        if plot:
            self.fig, self.ax = plt.subplots(figsize=(5,5))
    
    def step(self, action):
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
        u_pred, v_pred = self.apply_boundary(u_pred, v_pred, action)
        # solve for pressure
        pressure = self.solve_pressure(u_pred, v_pred, p_prev)
        dpdx, dpdy = central_difference_x(pressure, dx), central_difference_y(pressure, dy)
        u_next = u_pred - dt / DENSITY * dpdx
        v_next = v_pred - dt / DENSITY * dpdy
        u_next, v_next = self.apply_boundary(u_next, v_next, action)
        # TODO: wait reward function
        k = 2*np.pi
        u_target = -np.cos(k * self.x[:, np.newaxis]) * np.sin(k * self.y) * np.exp(-2 * 0.1 * k **2)
        v_target = np.sin(k * self.x[:, np.newaxis]) * np.cos(k * self.y) * np.exp(-2 * 0.1 * k**2)
        self.u_target = u_target
        self.v_target = v_target
        reward = - np.linalg.norm(u_next[1:-1]-u_target[1:-1]) - np.linalg.norm(v_next[1:-1]-v_target[1:-1])
        self.reward = reward
        self.u_prev, self.v_prev = u_next, v_next
        obs = np.stack([self.u_prev, self.v_prev], axis=-1)
        self.t += self.dt
        done = self.t >= self.T
        truncated = False
        info = {}
        return obs, reward, done, truncated, info

    def reset(self, seed=0):
        self.u_prev = np.zeros_like(self.X) # -np.cos(k * self.x[:, np.newaxis]) * np.sin(k * self.y)
        self.v_prev = np.zeros_like(self.X) # np.sin(k * self.x[:, np.newaxis]) * np.cos(k * self.y)
        self.p_prev =np.zeros_like(self.X) # -1/4 *  (np.cos(2 * k * self.x[:, np.newaxis]) + np.cos(2 * k * self.y))
        obs = np.stack([self.u_prev, self.v_prev], axis=-1)
        return obs, None

    def apply_boundary(self, u, v, action):
        #TODO: pass boundary control conditions
        u[0, :] =  action[:self.Nx] # lower
        u[:, 0] = action[self.Nx:self.Nx*2]  # left 
        u[:, -1] = action[self.Nx*2:self.Nx*3] # right
        u[-1, :] = action[self.Nx*3:self.Nx*4] # np.pi/d * np.cos(1.0/d) * np.sin(self.y/d) * np.exp(-2 * np.pi ** 2 * KINEMATIC_VISCOSITY * self.t / d / d) # action[self.Nx*3:self.Nx*4] # upper
        v[0, :] = action[self.Nx*4:self.Nx*5] # lower
        v[:, 0] =  action[self.Nx*5:self.Nx*6] # left
        v[:, -1] = action[self.Nx*6:self.Nx*7] # right
        v[-1, :] = action[self.Nx*7:self.Nx*8] # upper
        return u, v 
    

    def solve_pressure(self, u, v, p_prev):
        dudx = central_difference_x(u, self.dx)
        dvdy = central_difference_y(v, self.dy)
        rhs = DENSITY / self.dt * (dudx + dvdy)
        for _ in range(N_PRESSURE_POISSON_ITERATIONS):
            p_next = p_prev.copy()
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
            k = 2*np.pi
            p_next[:, -1] = p_next[:, -2] # -1/4 *np.exp(-4 * 0.1 * k**2 * env.t) * (np.cos(2 * k * self.x) + np.cos(2 * k * 1.0))
            p_next[0,  :] = p_next[1,  :]  #-1/4 *np.exp(-4 * 0.1 * k**2 * env.t) * (np.cos(2 * k * 0) + np.cos(2 * k * self.y))
            p_next[:,  0] = p_next[:,  1] # -1/4 *np.exp(-4 * 0.1 * k**2 * env.t) * (np.cos(2 * k * self.x) + np.cos(2 * k * 0))
            p_next[-1, :] = p_next[-2, :] # -1/4 *np.exp(-4 * 0.1 * k**2 * env.t) * (np.cos(2 * k * 1) + np.cos(2 * k * self.y))
            p_prev = p_next
        self.p = p_prev
        return p_prev
    
    def render(self):
        self.ax.clear()
        #a = self.u_prev**2 + self.v_prev**2
        self.ax.contourf(self.X, self.Y, self.p, cmap="coolwarm")
        self.ax.quiver(self.X, self.Y, self.u_prev, self.v_prev, color="black")
        #self.ax.quiver(self.X, self.Y, self.u_target, self.v_target, color="red")
        self.ax.set_ylim((0, np.max(self.x)))
        self.ax.set_ylim((0, np.max(self.y)))
        self.ax.set_title(f't = {np.round(self.t, 5)}, reward={np.round(self.reward, 2)}')
        plt.savefig(f'frames/frame_{int(self.t*1000)}.png', dpi=300)
        plt.pause(1e-3)
        



from gym.envs.registration import register
register(
    id='NSPDEEnv-v0',
    entry_point='NavierStokes:NSPDEEnv',
)

if __name__ == "__main__":
    import imageio
    if not os.path.exists('frames'):
        os.makedirs('frames')
    env = NSPDEEnv(T=0.2, dt=0.001, dx=0.05, dy=0.05, plot=True)
    rewards = 0.
    for i in range(200):
        action = np.zeros(8 * env.Nx)
        k = 2*np.pi
        action[:env.Nx] = -np.cos(k * 0.) * np.sin(k * env.y) * np.exp(-2 * 0.1 * k **2)
        action[env.Nx:env.Nx*2] = -np.cos(k * env.x) * np.sin(k * 0.) * np.exp(-2 * 0.1 * k **2)
        action[env.Nx*2:env.Nx*3] = -np.cos(k * env.x) * np.sin(k * 1.0) * np.exp(-2 * 0.1 * k **2 )
        action[env.Nx*3:env.Nx*4] = -np.cos(k * 1) * np.sin(k * env.y) * np.exp(-2 * 0.1 * k **2)
        action[env.Nx*4:env.Nx*5] =  np.sin(k * 0) * np.cos(k * env.y) * np.exp(-2 * 0.1 * k**2 )
        action[env.Nx*5:env.Nx*6] =  np.sin(k * env.x) * np.cos(k * 0) * np.exp(-2 * 0.1 * k**2)
        action[env.Nx*6:env.Nx*7] =  np.sin(k * env.x) * np.cos(k * 1) * np.exp(-2 * 0.1 * k**2 )
        action[env.Nx*7:env.Nx*8] =  np.sin(k * 1) * np.cos(k * env.y) * np.exp(-2 * 0.1 * k**2)
        obs, reward, done, t, info = env.step(action)
        env.render()
        rewards += reward
    print('rewards', rewards)
    frame_files = [f'frames/frame_{t}.png' for t in range(1,100)]
    frames = [imageio.imread(f) for f in frame_files]
    imageio.mimsave('../../examples/NavierStokes/animation.gif', frames, fps=10)
    import shutil
    shutil.rmtree('frames')