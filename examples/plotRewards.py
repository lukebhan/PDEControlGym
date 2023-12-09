import matplotlib.pyplot as plt
import numpy as np
from utils import set_size
from utils import load_csv

# PLOTS A SINGLE SET OF REWARDS AS A SINGLE LINE. 
# REPEAT THE CODE FOR DIFFERENT ALGORITHMS ON SAME PLOT

# Set your tensorboard avg_rew paths
filenames = ["./tbRew/PPO_24.csv", "./tbRew/PPO_25.csv", "./tbRew/PPO_26.csv", "./tbRew/PPO_27.csv", "./tbRew/PPO_28.csv"]

timeArr = []
rewardArr = []
for name in filenames:
    times, rewards = load_csv(name)
    timeArr.append(times)
    rewardArr.append(rewards)

# takes max amount of timesteps all data has
maxTimestep = np.inf
for data in timeArr:
    maxTimestep = min(maxTimestep, data[-1])
print(maxTimestep)

# remove data after minTimestep
maxDataSeq = []
for data in timeArr:
    for i in range(len(data)):
        if data[i] >= maxTimestep:
            maxDataSeq.append(i)
            break

print(maxDataSeq)

# Get mean and std of each value at time step 
rewardArrClean = []
for i, data in enumerate(rewardArr):
    rewardArrClean.append(data[:maxDataSeq[i]])
rewardArr = np.array(rewardArrClean)
meanArr = rewardArr.mean(axis=0)
stdArr = rewardArr.std(axis=0)

# Set size according to latex textwidth
fig = plt.figure(figsize=set_size(432, 0.99, (1, 1), height_add=0))

ax = fig.subplots(ncols=1)
t = timeArr[0]
x = t[:maxDataSeq[0]]
mean = meanArr
std = stdArr
# 95 confidence interval
cis = (mean - 2*std, mean + 2*std)
ax.plot(x, mean, label="PPO")
ax.fill_between(x, cis[0], cis[1], alpha=0.2)
plt.show()
