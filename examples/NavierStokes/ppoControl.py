import sys
sys.path.append('../../')
sys.path.append('../../pdeGym/envs/')
import NavierStokes
import gym
from stable_baselines3 import PPO
import numpy as np
env = gym.make("NSPDEEnv-v0", T=0.2, dx=0.05, dy=0.05)

model = PPO("MlpPolicy", env, verbose=1, gamma=1.0, tensorboard_log="./ppo_NS_tensorboard/")
model.learn(total_timesteps=500_000)
model.save("models/PPO")

# Test
# model = PPO("MlpPolicy", env, verbose=1, gamma=1.05, tensorboard_log="./ppo_NS_tensorboard/")
# model.load("models/PPO")
env = gym.make("NSPDEEnv-v0", T=0.2, dx=0.05, dy=0.05, plot=True)
obs, _ = env.reset()
done = False
rewards = 0.
t = 0
while not done:
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    rewards += reward
    t += 1
print(reward)
env.close()


