import gymnasium as gym
import numpy as np

from gym_quad.envs.mujoco import UAVQuadBase
from gym_quad.controller import RpyController


class Trajectory:
    R = 0.5     # trajectory radius
    w = 1.0     # trajectory angular speed (rad/s)

def main():

    env = gym.make('UAVQuadBase-v0', render_mode='human')

    dt = env.dt
    mass = env.mass
    gravity = env.gravity[2]

    # controller
    ctrl = RpyController(dt, mass, gravity=gravity)

    # [x, y, z, q0, q1, q2, q3]
    observation,_ = env.reset()
    # print(observation)
    for t in range(1000):

        env.render()

        # current position and quaternion
        s = np.array([observation[0], observation[1], observation[2]])
        q = observation[3:7]

        # desired trajectory (position and yaw)
        pos_d = np.array([
            Trajectory.R * np.cos(Trajectory.w * dt * t),
            Trajectory.R * np.sin(Trajectory.w * dt * t),
            1.0
        ])
        yaw_d = (Trajectory.w * dt * t + np.pi) % (2 * np.pi) - np.pi

        # control
        action = ctrl.control(s, q, pos_d, yaw_d)
        observation, reward, done,_,info = env.step(action)

        if done:
            break

if __name__ == "__main__":
    main()