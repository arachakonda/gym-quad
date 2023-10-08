from gymnasium.envs.registration import register

register(
    id="UAVQuadBase-v0",
    entry_point="gym_quad.envs.mujoco:UAVQuadBase",

)