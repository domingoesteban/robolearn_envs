# Robolearn Environments
>A Python package with OpenAI-Gym-like environments for Robot Learning

<table align="center">
    <tr>
    <td style="width:150px; height:150px; background-color:red;text-align:center; vertical-align:middle">
    <img src="docs/img/centauro_obst.png" alt="robolearn_logo" width="100" height="100" class="center" />
    </td>
    <td style="width:150px; height:150px; background-color:red;text-align:center; vertical-align:middle">
    <img src="docs/img/cogimon2.png" alt="robolearn_logo" width="100" height="100" class="center" />
    </td>
    <td style="width:150px; height:150px; background-color:red;text-align:center; vertical-align:middle">
    <img src="docs/img/coman.png" alt="robolearn_logo" width="100" height="100" class="center" />
    </td>
    <td style="width:150px; height:150px; background-color:red;text-align:center; vertical-align:middle">
    <img src="docs/img/walkman.png" alt="robolearn_logo" width="100" height="100" class="center" />
    </td>
    <td style="width:150px; height:150px; background-color:red;text-align:center; vertical-align:middle">
    <img src="docs/img/hyq.png" alt="robolearn_logo" width="100" height="100" class="center" />
    </td>
    </tr>
</table>

This repository contains a shorter version of the Pybullet-Environment interface
of **Robolearn**.

The code has been tested with Python 3.5 (or later).

**Robolearn** is a python package that defines common interfaces
between robot learning algorithms and robots. More info in the following
[link](https://github.com/domingoesteban/robolearn).

<p align="center">
<img src="docs/img/robolearn_logo2.png" alt="robolearn_logo" width="100" height="100" class="center" />
</p>

**Warning**: This package is gradually becoming public, so the public version is still in development. Sorry for any inconvenience.

## Pre-Installation
It is recommended to first create either a virtualenv or a conda environment.
- **Option 1: Conda environment**. First install either miniconda (recommended) or anaconda. 
[Installation instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
```bash
# Create the conda environment
conda create -n <condaenv_name> python=3.5
# Activate the conda environment
conda activate <condaenv_name>

```

- **Option 2: Virtualenv**. First install pip and virtualenv. 
[Installation instructions](https://packaging.python.org/guides/installing-using-pip-and-virtualenv/)
```bash
# Create the virtual environment
virtualenv -p python3.5 <virtualenv_name>
# Activate the virtual environment
source <virtualenv_name>/bin/activate
```

## Installation
1. Clone this repository
```bash
git clone https://github.com/domingoesteban/hiu_sac
```

2. Install the requirements of this repository
```bash
cd robolearn_envs
pip install -e .
```

## Use
Standard-OpenAI-Gym way:
```python
import gym
import robolearn_envs

env = gym.make('RoboLearn-CogimonLocomotionRender-v0')
# A 'headless' version of the same environment would be:
# env = gym.make('RoboLearn-CogimonLocomotion-v0')

# By default, Robolearn Bullet environments starts rendering when they are reset.
obs = env.reset()

for _ in range(500):
    obs, reward, done, info = env.step(env.action_space.sample())

env.close()
```

Easy-to-customize way:
```python
from robolearn_envs.pybullet import CogimonLocomotionEnv

env = CogimonLocomotionEnv(
    active_joints='WB',
    control_mode='joint_torque',
    is_render=True,
    sim_timestep=0.001,
    frame_skip=10,
    seed=1510,
    max_time=None,
)

# If is_render=True, rendering starts when the environment is reset.
obs = env.reset()

for _ in range(100):
    obs, reward, done, info = env.step(env.action_space.sample())

env.close()
```

## Environments
| Robot | Task | Gym Name  |
| ------------ |:------:|:-------:|
|  |  |   |
| Centauro | Reaching | CentauroReachingEnv-v0  |
| Centauro | Locomotion | CentauroLocomotionEnv-v0  |
|  |  |   |
| Cogimon | Reaching | CogimonReachingEnv-v0  |
| Cogimon | Locomotion | CogimonLocomotionEnv-v0  |
|  |  |   |
| Walkman | Reaching | WalkmanReachingEnv-v0  |
| Walkman | Locomotion | WalkmanLocomotionEnv-v0  |
|  |  |   |
| Coman | Locomotion | ComanLocomotionEnv-v0  |
|  |  |   |
| Hyq | Locomotion | HyqLocomotionEnv-v0  |


## Citation
If this repository was useful for your research, we would appreciate that you can cite it:

    @misc{robolearn-envs,
      author = {Esteban, Domingo},
      title = {RoboLearn Environments},
      year = {2018},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/domingoesteban/robolearn_envs}},
    }


## Acknowledgements
- *Erwin Coumans* for Bullet, PyBullet, and his didactic examples 
([Bullet repository](https://github.com/bulletphysics/bullet3))
- *Enrico Mingo* for
[Coman](https://github.com/ADVRHumanoids/iit-coman-ros-pkg),
[Walkman](https://github.com/ADVRHumanoids/iit-walkman-ros-pkg), and
[Cogimon](https://github.com/ADVRHumanoids/iit-cogimon-ros-pkg)
urdf models.
- *Malgorzata Kamedula* for
[Centauro](https://github.com/ADVRHumanoids/centauro-simulator/tree/master/centauro_gazebo)
urdf model.
- *Dynamic Legged Systems Lab* for
[HyQ](https://github.com/iit-DLSLab/hyq-description) urdf model.
