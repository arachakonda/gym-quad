import os
import gymnasium as gym
from gym_quad.envs.mujoco import UAVQuadHover

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make('UAVQuadHover-v0')
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
env = DummyVecEnv([lambda: env])

log_path = os.path.join('Training', 'Logs')

model = PPO('MlpPolicy', env, verbose=0, tensorboard_log=log_path)
model.learn(total_timesteps=30000000, progress_bar=True)

PPO_Path = os.path.join('Training', 'Models', 'PPO_quad')
model.save(PPO_Path)



evaluate_policy(model, env, n_eval_episodes=10, render=False)
