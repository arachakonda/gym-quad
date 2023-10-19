from gymnasium.envs.registration import register

register(
    id="UAVQuadBase-v0",
    entry_point="gym_quad.envs.mujoco:UAVQuadBase",

)

register(
    id="UAVQuadBaseSlung-v0",
    entry_point="gym_quad.envs.mujoco:UAVQuadBaseSlung",

)

register(
    id="UAVQuadHover-v0",
    entry_point="gym_quad.envs.mujoco:UAVQuadHover",

)