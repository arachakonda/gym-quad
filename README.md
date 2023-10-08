# gym-quad

`gym-quad` is a Gym-based package for quadcopter simulations, optimized for `mujoco>=2.3.3` and the latest Gym releases. It is based on ETH Zurich's reinmav-gym package

### Features:

- Compatibility with `mujoco>=2.3.3`.
- Designed for extensibility.

### Installation:

Clone the repository and install:

```bash
git clone https://github.com/yourusername/gym-quad.git
cd gym-quad
pip install -e .
```

### Usage:

```python
import gym
env = gym.make("UAVQuadBase-v0")
```

For further details, refer to the Gymnasium documentation.

(Note: I created this repo for personal practice)