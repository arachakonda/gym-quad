import argparse
import os
import gymnasium as gym

from stable_baselines3.common.envs import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env import VecVideoRecorder, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO
from gym_quad.envs.mujoco import UAVQuadBase

def parse_args():
    parser = argparse.ArgumentParser()

    # ... [rest of the arguments] ...

    return parser.parse_args()

def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    return _init

def train(args):
    # ... [rest of the train function code, adjusted for SB3] ...
    pass

def build_env(args):
    env_id = 'MujocoQuadForce-v1'
    seed = args.seed or 0
    num_env = args.num_env or 1

    if num_env == 1:
        env = DummyVecEnv([make_env(env_id, i, seed) for i in range(num_env)])
    else:
        env = SubprocVecEnv([make_env(env_id, i, seed) for i in range(num_env)], start_method='fork')
    
    # Use VecNormalize for normalizing observations and rewards
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    return env

def main():
    # ... [rest of the main function code, adjusted for SB3] ...
    pass

if __name__ == '__main__':
    main()