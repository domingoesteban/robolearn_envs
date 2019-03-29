"""
Based on PyBullet Gym examples file: scene_abstract.py

https://github.com/bulletphysics/bullet3
"""

from __future__ import print_function
import pybullet as pb
import gym
import numpy as np


class BulletWorld(object):
    """
    A base class for a world. It manages the physics properties of pybullet.
    """

    def __init__(self, gravity=9.8, timestep=1/240., frame_skip=1,
                 solver_iter=50, np_random=None, pybullet_client=None):
        if np_random is None:
            self.np_random, seed = gym.utils.seeding.np_random(None)
        else:
            self.np_random = np_random
        # Simulation properties
        self.sim_time_step = timestep
        self._frame_skip = frame_skip
        self.solver_iter = solver_iter
        self._is_real_time = False

        # Physics
        self.gravity = gravity

        if pybullet_client is None:
            pybullet_client = pb

        self.pbc = pybullet_client

        self.reset_world()

    @property
    def pbc(self):
        return self._pybullet_client

    @pbc.setter
    def pbc(self, pb_client):
        self._pybullet_client = pb_client

    def reset_world(self):
        """
        This function gets overridden by specific scene, to reset specific
        objects into their start positions.
        """
        self.pbc.resetSimulation()

        self.configure_world()

        self.reset_world_time()

    def configure_world(self):
        self.pbc.setGravity(0, 0, -self.gravity)
        # pb.setPhysicsEngineParameter(fixedTimeStep=self.timestep*self.frame_skip,
        self.pbc.setPhysicsEngineParameter(fixedTimeStep=self.sim_time_step,
                                           numSolverIterations=self.solver_iter,
                                           numSubSteps=0)
        # numSubSteps=1)
        # numSubSteps=self.frame_skip)

        self.pbc.setTimeStep(self.sim_time_step)

    def world_step(self):
        """
        The idea is: apply motor torques for all robots, then call
        world_step(), then collect observations from robots using step() with
        the same action.
        """
        for _ in range(self._frame_skip):
            self.pbc.stepSimulation()

        self._accum_sim_steps += 1

    def reset_world_time(self):
        self._accum_sim_steps = 0

    @property
    def dt(self):
        return self.sim_time_step * self._frame_skip

    @property
    def sim_time_step(self):
        return self._time_step

    @property
    def sim_time(self):
        return self._accum_sim_steps * self.dt

    @sim_time_step.setter
    def sim_time_step(self, time_step):
        self._time_step = time_step

    def get_physics_parameters(self):
        params = self.pbc.getPhysicsEngineParameters()
        physics_params = {
            'gravity': np.array([
                params['gravityAccelerationX'],
                params['gravityAccelerationY'],
                params['gravityAccelerationZ'],
            ]),
            'timestep': params['fixedTimeStep'],
            'num_substeps': params['numSubSteps'],
            'num_solver_iters': params['numSolverIterations'],
        }

        return physics_params

    def set_real_time(self, enable=True):
        self.pbc.setRealTimeSimulation(enableRealTimeSimulation=int(enable))
        self._is_real_time = enable

    @property
    def is_real_time(self):
        return self._is_real_time
