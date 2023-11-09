import sys
sys.path.append('../../')
sys.path.append('../../pdeGym/envs/')
import NavierStokes
import gym
from stable_baselines3 import PPO

env = gym.make("NSPDEEnv-v0", T=0.2)

model = PPO("MlpPolicy", env, verbose=1, gamma=1.01, tensorboard_log="./ppo_NS_tensorboard/")
model.learn(total_timesteps=10_000)
model.save("models/PPO")

# Test
env = gym.make("NSPDEEnv-v0", T=0.2, plot=True)
obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
env.close()