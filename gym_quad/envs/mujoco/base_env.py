import os
import math
from abc import ABC
from typing import Tuple, Dict
import numpy as np
from gymnasium import spaces
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 2.04,
}

class UAVQuadBase(MujocoEnv, utils.EzPickle):
    metadata = {"render_modes":["human", "rgb_array","depth_array"], "render_fps":50}
    def __init__(self, 
                 xml_file="quadrotor_ground.xml", 
                 frame_skip: int = 2, 
                 reset_noise_scale: float = 0.01, 
                 **kwargs,
                 ):
        xml_path = os.path.join(os.path.dirname(__file__), "./assets", xml_file)
        utils.EzPickle.__init__(self, frame_skip, reset_noise_scale, **kwargs)
        observation_space = spaces.Box(low = -np.inf, high=np.inf, shape=(13,), dtype=np.float64)
        self._reset_noise_scale = reset_noise_scale
        MujocoEnv.__init__(self,
                           xml_path,
                           frame_skip,
                           observation_space=observation_space,
                           default_camera_config=DEFAULT_CAMERA_CONFIG,
                           **kwargs)
        
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0/self.dt))
        }
        
    def step(self,action):
        self.do_simulation(self.clip_action(action),self.frame_skip)
        observation = self._get_obs()
        terminated = bool(not np.isfinite(observation).all())
        reward = 0
        info ={}

        if self.render_mode == "human":
            self.render()
        
        return observation,reward,terminated,False,info
    

    def clip_action(self,action):
        action = np.clip(action, a_min = 0, a_max=np.inf)
        return action
    
    def reset_model(self):
        # noise_low = -self._reset_noise_scale
        # noise_high = self._reset_noise_scale

        qpos = self.init_qpos
        qvel = self.init_qvel

        # qpos = self.init_qpos + self.np_random.uniform(
        #     size=self.model.nq, low=noise_low, high=noise_high
        # )
        # qvel = self.init_qvel + self.np_random.uniform(
        #     size=self.model.nv, low=noise_low, high=noise_high
        # )
        self.set_state(qpos, qvel)
        return self._get_obs()
    
    def _get_obs(self):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        self.mujoco_qpos = np.array(qpos)
        self.mujoco_qvel = np.array(qvel)

        obs = np.concatenate([qpos, qvel]).flatten()
        return obs
    
    @property
    def mass(self):
        return self.model.body_mass[1]

    @property
    def gravity(self):
        return self.model.opt.gravity
        
