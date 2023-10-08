from setuptools import setup

extras = {
    'mujoco': [],
}

# dependency
all_dependencies = []
for group_name in extras:
    all_dependencies += extras[group_name]
extras['all'] = all_dependencies

setup(name="gym_quad",
      version="0.0.1",
      url='https://arachakonda.github.io/gym-quad/',
      install_requires=[
          'matplotlib',
          'scipy',
          'numpy',
          'gymnasium',
          'mujoco',
          'imageio',],
        packages=['gym_quad',],
        extras_require=extras,
    
      )