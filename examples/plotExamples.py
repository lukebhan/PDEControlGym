import matplotlib.pyplot as plt
import numpy as np
import math
from utils import set_size
from utils import linestyle_tuple

u5PPO = np.loadtxt("data/UPPO5.txt")
u6PPO = np.loadtxt("data/UPPO6.txt")
u5BCK = np.loadtxt("data/UBCK5.txt")
u6BCK = np.loadtxt("data/UBCK6.txt")
u6SAC = u6PPO
u5SAC = u5PPO
print(u5BCK)
print(u6BCK)

fig = plt.figure(figsize=set_size(433, 0.99, (2, 3), height_add=1))
subfigs = fig.subfigures(nrows=2, ncols=1, hspace=0)

subfig = subfigs[0]
subfig.suptitle(r"Example trajectories for $u(0, x)=5$ with backstepping, PPO, and SAC")
subfig.subplots_adjust(left=0.03, bottom=0.05, right=1, top=0.95, wspace=0, hspace=0)
X = 1
dx = 1e-2
T = 10
spatial = np.linspace(dx, X, int(round(X/dx)))
temporal = np.linspace(0, T, len(u5PPO))
meshx, mesht = np.meshgrid(spatial, temporal)

ax = subfig.subplots(nrows=1, ncols=3, subplot_kw={"projection": "3d", "computed_zorder": False})

for axes in ax:
    for axis in [axes.xaxis, axes.yaxis, axes.zaxis]:
        axis._axinfo['axisline']['linewidth'] = 1
        axis._axinfo['axisline']['color'] = "b"
        axis._axinfo['grid']['linewidth'] = 0.2
        axis._axinfo['grid']['linestyle'] = "--"
        axis._axinfo['grid']['color'] = "#d1d1d1"
        axis.set_pane_color((1,1,1))

ax[0].plot_surface(meshx, mesht, u5BCK, edgecolor="black",lw=0.2, rstride=50, cstride=1, 
                        alpha=1, color="white", shade=False, rasterized=True, antialiased=True)
ax[0].view_init(10, 35)
ax[0].set_xlabel("x", labelpad=-3)
ax[1].set_xlabel("x", labelpad=-3)
ax[2].set_xlabel("x", labelpad=-3)
ax[0].set_ylabel("Time", labelpad=-3)
ax[2].set_ylabel("Time", labelpad=-3)
ax[1].set_ylabel("Time", labelpad=-3)
ax[0].set_zlabel(r"$u(x, t)$", rotation=90, labelpad=-7)

ax[0].zaxis.set_rotate_label(False)
ax[0].set_xticks([0, 0.5, 1])
ax[0].tick_params(axis='x', which='major', pad=-3)
ax[1].tick_params(axis='x', which='major', pad=-3)
ax[2].tick_params(axis='x', which='major', pad=-3)
ax[0].tick_params(axis='y', which='major', pad=-3)
ax[1].tick_params(axis='y', which='major', pad=-3)
ax[2].tick_params(axis='y', which='major', pad=-3)
ax[0].tick_params(axis='z', which='major', pad=-1)
ax[1].tick_params(axis='z', which='major', pad=-1)
ax[2].tick_params(axis='z', which='major', pad=-1)
test = np.ones(len(temporal))
vals = (u5BCK.transpose())[-1] 
ax[0].plot(test[1:], temporal[1:], vals[1:], color="red", lw=0.1, antialiased=False, rasterized=False)
 
ax[1].plot_surface(meshx, mesht, u5PPO, edgecolor="black",lw=0.2, rstride=50, cstride=1, 
                        alpha=1, color="white", shade=False, rasterized=True, antialiased=True)
ax[1].view_init(10, 35)
ax[1].zaxis.set_rotate_label(False)
ax[1].set_xticks([0, 0.5, 1])
test = np.ones(len(temporal))
vals = (u5PPO.transpose())[-1] 
ax[1].plot(test[1:], temporal[1:], vals[1:], color="red", lw=0.1, antialiased=False, rasterized=False)
 
ax[2].plot_surface(meshx, mesht, u5PPO, edgecolor="black",lw=0.2, rstride=50, cstride=1, 
                        alpha=1, color="white", shade=False, rasterized=True, antialiased=True)
ax[2].view_init(10, 35)
ax[2].zaxis.set_rotate_label(False)
ax[2].set_xticks([0, 0.5, 1])
ax[0].set_zticks([-10, 0, 10])
test = np.ones(len(temporal))
vals = (u5PPO.transpose())[-1] 
ax[2].plot(test[1:], temporal[1:], vals[1:], color="red", lw=0.1, antialiased=False, rasterized=False)
 
subfig = subfigs[1]
subfig.suptitle(r"Example trajectories for $u(0, x)=6$ with backstepping, PPO, and SAC")
X = 1
dx = 1e-2
T = 10
spatial = np.linspace(dx, X, int(round(X/dx)))
temporal = np.linspace(0, T, len(u5PPO))
meshx, mesht = np.meshgrid(spatial, temporal)

ax = subfig.subplots(nrows=1, ncols=3, subplot_kw={"projection": "3d", "computed_zorder": False})
subfig.subplots_adjust(left=0.03, bottom=0.05, right=1, top=0.95, wspace=0, hspace=0)
for axes in ax:
    for axis in [axes.xaxis, axes.yaxis, axes.zaxis]:
        axis._axinfo['axisline']['linewidth'] = 1
        axis._axinfo['axisline']['color'] = "b"
        axis._axinfo['grid']['linewidth'] = 0.2
        axis._axinfo['grid']['linestyle'] = "--"
        axis._axinfo['grid']['color'] = "#d1d1d1"
        axis.set_pane_color((1,1,1))

ax[0].plot_surface(meshx, mesht, u6BCK, edgecolor="black",lw=0.1, rstride=50, cstride=1, 
                        alpha=1, color="white", shade=False, rasterized=True, antialiased=True)
ax[0].view_init(10, 35)
ax[0].set_xlabel("x", labelpad=-3)
ax[1].set_xlabel("x", labelpad=-3)
ax[2].set_xlabel("x", labelpad=-3)
ax[0].set_ylabel("Time", labelpad=-3)
ax[2].set_ylabel("Time", labelpad=-3)
ax[1].set_ylabel("Time", labelpad=-3)
ax[0].set_zlabel(r"$u(x, t)$", rotation=90, labelpad=-7)

ax[0].zaxis.set_rotate_label(False)
ax[0].set_xticks([0, 0.5, 1])
ax[0].tick_params(axis='x', which='major', pad=-3)
ax[1].tick_params(axis='x', which='major', pad=-3)
ax[2].tick_params(axis='x', which='major', pad=-3)
ax[0].tick_params(axis='y', which='major', pad=-3)
ax[1].tick_params(axis='y', which='major', pad=-3)
ax[2].tick_params(axis='y', which='major', pad=-3)
ax[0].tick_params(axis='z', which='major', pad=-1)
ax[1].tick_params(axis='z', which='major', pad=-1)
ax[2].tick_params(axis='z', which='major', pad=-1)
ax[0].zaxis.set_rotate_label(False)
ax[0].set_xticks([0, 0.5, 1])
test = np.ones(len(temporal))
vals = (u6BCK.transpose())[-1] 
ax[0].plot(test[1:], temporal[1:], vals[1:], color="red", lw=0.1, antialiased=False, rasterized=False)
 
ax[1].plot_surface(meshx, mesht, u6PPO, edgecolor="black",lw=0.2, rstride=50, cstride=1,
                        alpha=1, color="white", shade=False, rasterized=True, antialiased=True)
ax[1].view_init(10, 35)
ax[1].zaxis.set_rotate_label(False)
ax[1].set_xticks([0, 0.5, 1])
test = np.ones(len(temporal))
vals = (u6PPO.transpose())[-1] 
ax[1].plot(test[1:], temporal[1:], vals[1:], color="red", lw=0.1, antialiased=False, rasterized=False)
ax[2].plot_surface(meshx, mesht, u6SAC, edgecolor="black",lw=0.1, rstride=50, cstride=1, 
                        alpha=1, color="white", shade=False, rasterized=True, antialiased=True)
ax[2].view_init(10, 35)
ax[2].zaxis.set_rotate_label(False)
ax[2].set_xticks([0, 0.5, 1])
test = np.ones(len(temporal))
vals = (u6SAC.transpose())[-1] 
ax[2].plot(test[1:], temporal[1:], vals[1:], color="red", lw=0.1, antialiased=False, rasterized=False)
 
plt.savefig("hyperbolicExamples.pdf", dpi=300)

# BUILD CONTROL SIGNAL PLOTS
fig = plt.figure(figsize=set_size(433, 0.99, (1, 2), height_add=1))
subfigs = fig.subfigures(nrows=1, ncols=1, hspace=0)

subfig = subfigs
subfig.suptitle(r"Control Signals for $u(0, x)=5$ and $u(0, x)=6$")
subfig.subplots_adjust(left=0.1, bottom=0.2, right=.98, top=0.86, wspace=0.25, hspace=0.1)
X = 1
dx = 1e-2
T = 10
spatial = np.linspace(dx, X, int(round(X/dx)))
temporal = np.linspace(0, T, len(u5PPO))
ax = subfig.subplots(nrows=1, ncols=2)
l1, = ax[0].plot(temporal, u5PPO.transpose()[-1], label="PPO", linestyle=linestyle_tuple[2][1], color="orange")
l2, = ax[0].plot(temporal, u6SAC.transpose()[-1], label="SAC", linestyle=linestyle_tuple[2][1], color="green")
l3, = ax[0].plot(temporal, u5BCK.transpose()[-1], label="Backstepping", color="#0096FF")
ax[0].set_xlabel("Time")
ax[0].set_ylabel(R"$U(t)$", labelpad=-2)

l1, = ax[1].plot(temporal, u6PPO.transpose()[-1], label="PPO", linestyle=linestyle_tuple[2][1], color="orange")
l2, = ax[1].plot(temporal, u5SAC.transpose()[-1], label="SAC", linestyle=linestyle_tuple[2][1], color="green")
l3, = ax[1].plot(temporal, u6BCK.transpose()[-1], label="Backstepping", color="#0096FF")
ax[1].set_xlabel("Time")
ax[1].set_ylabel(r"$U(t)$", labelpad=-2)
plt.legend([l1, l2, l3], ["PPO", "SAC", "Backstepping"], loc="lower left", bbox_to_anchor=[.56,.86])
plt.savefig("hyperbolicControlExamples.pdf", dpi=300)
