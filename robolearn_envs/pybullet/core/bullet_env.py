"""
Based on PyBullet Gym examples file: env_bases.py
https://github.com/bulletphysics/bullet3
"""

from __future__ import print_function
import numpy as np
import pybullet as pb
from robolearn_envs.core.robolearn_env import RobolearnEnv
from robolearn_envs.pybullet.core.bullet_world import BulletWorld
from robolearn_envs.pybullet.core.bullet_robot import BulletMultibody
from robolearn_envs.pybullet.core.bullet_client import BulletClient
from robolearn_envs.pybullet.core.bullet_robot import BulletRobot
from robolearn_envs.pybullet.core.bullet_object import BulletObject
import warnings
import os


class BulletEnv(RobolearnEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
    }

    def __init__(self, gravity=9.8, sim_timestep=1/240., frameskip=1,
                 is_render=False, seed=None):
        """
        Bullet Environment Class
        Args:
            gravity (float): Gravity
            sim_timestep (float):
            frameskip (int):
            is_render (bool):
        """
        RobolearnEnv.__init__(self, seed=seed)

        # Pybullet World
        self._world = None
        self.gravity = gravity
        self.sim_ts = sim_timestep
        self.frame_skip = frameskip

        self._sim_multibodies = {}

        # Rendering RGB
        self._render_data = {
            'distance': 3,
            'yaw': 0,  # 0
            'pitch': -30,  # -30
            'width': 320,
            'height': 320,  # 240
            'target_pos': [0, 0, 0]
        }

        # Rendering Human (Visualizer)
        self._vis_data = {
            'distance': 3.5,
            'yaw': 40,
            'pitch': -40,
            'target_pos': [0, 0, 0]
        }
        self._is_rendering = is_render
        self._pb_client = None
        self.add_pbc(is_render)

        self.add_world()

        # Seed
        if not hasattr(self, 'np_random'):
            self.np_random = None
            self.seed(None)

        # Others
        self._log_video = None

    @property
    def dt(self):
        return self.world.dt

    @property
    def env_time(self):
        return self.step_counter * self.dt

    @property
    def sim_time(self):
        return self.world.sim_time

    def reset_simulation(self, *args, **kwargs):
        self.disable_vis()

        if self._world is None:
            self.add_world(reset_time=True)

        # Restart World
        self._world.reset_world()

        if self._is_rendering:
            self.reset_visualizer_camera()

        self.update_pbcs()

        # # Reset bodies/objects
        # self.reset_multibodies()

    # ####################### #
    # Environment Multibodies #
    # ####################### #

    def add_to_sim(self, multibody, name=None):
        """Add multibody to the current environment.
        Args:
            multibody (BulletMultibody): Multibody to add to the environment
            name (str or None): Name of the multibody

        Returns:
            None

        """
        if not isinstance(multibody, BulletMultibody):
            raise TypeError("Multibody is not a %s instance."
                            % BulletMultibody.__name__)
        multibody.pbc = self.pbc
        if name is None:
            name = multibody.name + str('..00')
        while name in self._sim_multibodies.keys():
            base_name, name_number = name.split("..", 1)
            name = base_name + str('..%02d' % (int(name_number) + 1))

        self._sim_multibodies[name] = multibody

    def reset_multibodies(self):
        for multibody in self._sim_multibodies.values():
            multibody.reset()

    def spawn_multibodies(self):
        for multibody in self._sim_multibodies.values():
            multibody.spawn_in_pb()

    def get_robots(self):
        """Get the robots in the current environment

        Returns:
            list of BulletRobot: List of robots

        """
        return [multibody for multibody in self._sim_multibodies.values()
                if isinstance(multibody, BulletRobot)]

    def get_objects(self):
        """Get the objects (underactuated multibodies) in the current environment

        Returns:
            list of BulletObject: List of objects

        """
        return [multibody for multibody in self._sim_multibodies.values()
                if isinstance(multibody, BulletObject)]

    # ############### #
    # Pybullet Client #
    # ############### #

    @property
    def pbclient(self):
        return self._pb_client

    @pbclient.setter
    def pbclient(self, pb_client):
        self._pb_client = pb_client
        self.update_pbcs()

    pbc = pbclient

    def add_pbc(self, rendering=False):
        # self._is_rendering = rendering
        # Pybullet Client
        if self._is_rendering:
            self._pb_client = BulletClient(connection_mode=pb.GUI)
            self._pb_client.configureDebugVisualizer(
                self._pb_client.COV_ENABLE_GUI,
                0
            )
            # In case machine does not support opengl3
            # self._pb_client = BulletClient(connection_mode=pb.GUI,
            #                                      options="--opengl2")
            self.disable_vis()
        else:
            self._pb_client = BulletClient(connection_mode=pb.DIRECT)

    def update_pbcs(self):
        if self.world:
            self.world.pbc = self.pbc

        for multibody in self._sim_multibodies.values():
            multibody.pbc = self.pbc

    def remove_pbcs(self):
        if self.world:
            self.world.pbc = None
        for multibody in self._sim_multibodies.values():
            multibody.pbc = None

    # ##### #
    # World #
    # ##### #

    @property
    def world(self):
        return self._world

    def add_world(self, world=None, reset_time=True):
        if world is None:
            self._world = BulletWorld(gravity=self.gravity,
                                      timestep=self.sim_ts,
                                      frame_skip=self.frame_skip,
                                      pybullet_client=self.pbc,
                                      )
        else:
            if not isinstance(world, BulletWorld):
                raise TypeError("The world is not a %r class." % BulletWorld)
            self._world = world

        if reset_time:
            self.world.reset_world_time()

        self.update_pbcs()

    def sim_step(self):
        self._world.world_step()

    render_warn_once = True

    def render(self, mode='human', close=False):
        if mode == 'human':
            if self._is_rendering is False:
                if not self.render_warn_once:
                    return
                else:
                    warnings.warn("Human rendering deactivated in headless PyBulletEnv.")
                    return
                # self.set_rendering(True)
            # self._is_rendering = True

        if mode == 'rgb_array':
            return self.get_bullet_rgb()

        else:
            return np.array([])

    def __del__(self):
        if hasattr(self, '_pb_client') and self._pb_client is not None:
            self._pb_client.disconnect()

    def set_visualizer_data(self, distance=None, yaw=None, pitch=None,
                            target_pos=None):
        """
        Update the rendering data for the GUI.
        Args:
            distance:
            yaw:
            pitch:
            target_pos:

        Returns:

        """
        if distance is not None:
            self._vis_data['distance'] = distance
        if yaw is not None:
            self._vis_data['yaw'] = yaw
        if pitch is not None:
            self._vis_data['pitch'] = pitch
        if target_pos is not None:
            self._vis_data['target_pos'] = target_pos

    def set_render_data(self, distance=None, yaw=None, pitch=None,
                        target_pos=None, width=None, height=None):
        """
        Update the rendering data.
        Args:
            distance:
            yaw:
            pitch:
            target_pos:
            width:
            height:

        Returns:

        """
        if distance is not None:
            self._render_data['distance'] = distance
        if yaw is not None:
            self._render_data['yaw'] = yaw
        if pitch is not None:
            self._render_data['pitch'] = pitch
        if target_pos is not None:
            self._render_data['target_pos'] = target_pos
        if width is not None:
            self._render_data['width'] = int(width)
        if height is not None:
            self._render_data['height'] = int(height)

    def get_render_data(self):
        return self._pb_client.getDebugVisualizerCamera()

    def reset_visualizer_camera(self):
        self.pbc.resetDebugVisualizerCamera(
            cameraDistance=self._vis_data['distance'],
            cameraYaw=self._vis_data['yaw'],
            cameraPitch=self._vis_data['pitch'],
            cameraTargetPosition=self._vis_data['target_pos']
        )

    def enable_vis(self):
        if self._is_rendering:
            self.pbc.configureDebugVisualizer(
                self.pbc.COV_ENABLE_RENDERING,
                1
            )
            self.pbc.configureDebugVisualizer(
                self.pbc.COV_ENABLE_TINY_RENDERER,
                1
            )

    def disable_vis(self):
        self.pbc.configureDebugVisualizer(self.pbc.COV_ENABLE_GUI, 0)

        # TODO: 'BORRAR CONFIGURE DEBUG VISUALIZER'
        self.pbc.configureDebugVisualizer(self.pbc.COV_ENABLE_WIREFRAME, 0)
        self.pbc.configureDebugVisualizer(self.pbc.COV_ENABLE_SHADOWS, 0)

        # Temporally deactivate visualizer (to improve performance)
        self.pbc.configureDebugVisualizer(self.pbc.COV_ENABLE_RENDERING, 0)
        self.pbc.configureDebugVisualizer(self.pbc.COV_ENABLE_TINY_RENDERER, 0)

        # self.pbc.configureDebugVisualizer(
        #     pb_client.COV_ENABLE_PLANAR_REFLECTION, 0
        # )
        # self.pbc.configureDebugVisualizer(
        #     pb_client.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1
        # )

    def get_bullet_rgb(self):
        view_matrix = self._pb_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self._render_data['target_pos'],
            distance=self._render_data['distance'],
            yaw=self._render_data['yaw'],
            pitch=self._render_data['pitch'],
            roll=0,
            upAxisIndex=2)

        width = self._render_data['width']
        height = self._render_data['height']

        proj_matrix = self._pb_client.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(width)/height,
            nearVal=0.1, farVal=100.0)

        (_, _, px, _, _) = self._pb_client.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            # renderer=self._pb_client.ER_TINY_RENDERER
            renderer=self._pb_client.ER_BULLET_HARDWARE_OPENGL
        )

        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]

        return rgb_array

    def set_rendering(self, option):
        self._is_rendering = option

        if self._is_rendering and self.pbc.is_direct:
            self.pbc.saveBullet("state.bullet")
            # self.remove_pbcs()
            if self.pbc:
                self.close()
            # self.pbc = BulletClient(connection_mode=pb.GUI)
            self.add_pbc()
            self.reset_visualizer_camera()
            # self.add_world(reset_time=False)
            # if self._world is not None:
            #     self.world.pbc = self._pb_client
            # self.world.reset_world()
            # self.update_pbcs()
            # self.reset_multibodies()
            self.pbc.loadBullet("state.bullet")
            self.spawn_multibodies()
            os.remove("state.bullet")

        elif not self._is_rendering and self.pbc.is_gui:
            self.pbc.saveBullet("state.bullet")
            if self.pbc:
                self.close()
            self.pbc = BulletClient(connection_mode=pb.DIRECT)
            # self.add_world(reset_time=False)
            # if self._world is not None:
            #     self.world.pbc = self._pb_client
            # self.world.reset_world()

        # if update_all:
        #     self.update_pbcs()
        #     self.reset_multibodies()
        #     self.pbc.loadBullet("state.bullet")
        #     os.remove("state.bullet")

        # if self._is_rendering and self._pb_client.is_direct:
        #     self.pbc.saveBullet("state.bullet")
        #     del self._pb_client
        #     self._pb_client = BulletClient(connection_mode=pb.GUI)
        #     self.world.pbc = self.pbc
        #     # self.world.reset_world()
        #     self.pbc.loadBullet("state.bullet")
        # elif not self._is_rendering and self._pb_client.is_gui:
        #     self.pbc.saveBullet("state.bullet")
        #     del self._pb_client
        #     self._pb_client = BulletClient(connection_mode=pb.DIRECT)
        #     self.world.pbc = self.pbc
        #     # self.add_world(reset_time=False)
        #     self.pbc.loadBullet("state.bullet")

    def get_contacts(self, multibody1, multibody2=None,
                     link1=None, link2=None):
        """

        Args:
            multibody1 (BulletMultibody):
            multibody2 (BulletMultibody):
            link1 (Link):
            link2 (Link):

        Returns:

        """
        fcn_kwargs = {}
        if isinstance(multibody1, int):
            fcn_kwargs['bodyA'] = multibody1
        else:
            fcn_kwargs['bodyA'] = multibody1.id

        if multibody2 is not None:
            if isinstance(multibody1, int):
                fcn_kwargs['bodyB'] = multibody2
            else:
                fcn_kwargs['bodyB'] = multibody2.id
        if link1 is not None:
            if isinstance(link1, int):
                link1_id = link1
            else:
                print('--->', link1)
                link1_id = link1.id
            fcn_kwargs['linkIndexA'] = link1_id
        if link2 is not None:
            if isinstance(link2, int):
                link2_id = link2
            else:
                link2_id = link2.id
            fcn_kwargs['linkIndexB'] = link2_id

        # # Ugly hack because it is only calculated over the last sim step
        # self.pbc.stepSimulation()
        # temp_sim_state = self.pbc.saveState()
        contact_info = self.pbc.getContactPoints(**fcn_kwargs)
        # self.pbc.restoreState(stateId=temp_sim_state)

        # TODO: Return only useful data
        return contact_info

    def get_external_forces(self, multibody1, multibody2=None,
                            link1=None, link2=None):
        contacts = self.get_contacts(multibody1, multibody2, link1, link2)

        positions = np.array([cc[6] for cc in contacts])
        norm_forces = np.array([cc[9] for cc in contacts]).reshape(-1, 1)

        # (world frame) net force in links
        net_force = norm_forces * positions / (np.sum(norm_forces) + 1e-6)
        net_force = np.sum(net_force, axis=0)
        net_force = np.linalg.norm(net_force)

        return net_force

    def close(self):
        if self._pb_client:
            self._pb_client.disconnect()
        del self._pb_client
        self._pb_client = None

    def start_recording_video(self, file_name=None):
        if file_name is None:
            file_name = 'robolearn_bullet_env.mp4'
        self._log_video = self._pb_client.startStateLogging(
            self._pb_client.STATE_LOGGING_VIDEO_MP4, file_name
        )

    def stop_recording_video(self):
        if self._log_video is None:
            raise AttributeError("No video is being recorded!")
        self._pb_client.stopStateLogging(self._log_video)
        print("Stopping video...")
        self._log_video = None

    def dump_state_to_file(self, file_name):
        file = open(file_name, "w")

        for i in range(self.pbc.getNumBodies()):
            pos, orn = self.pbc.getBasePositionAndOrientation(i)
            lin_vel, ang_vel = self.pbc.getBaseVelocity(i)
            txt_pos = "pos=" + str(pos) + "\n"
            txt_orn = "orn=" + str(orn) + "\n"
            txt_lin_vel = "lin_vel" + str(lin_vel) + "\n"
            txt_ang_vel = "ang_vel" + str(ang_vel) + "\n"
            file.write(txt_pos)
            file.write(txt_orn)
            file.write(txt_lin_vel)
            file.write(txt_ang_vel)

        file.close()

    def debug(self, pause=True):
        keys = self.pbc.getKeyboardEvents()
        if pause:
            space_key = ord(' ')
            if space_key in keys and keys[space_key] & self.pbc.KEY_WAS_TRIGGERED:
                print("Simulation Paused!")
                print("Press Space key to start again!")
                while True:
                    keys = pb.getKeyboardEvents()
                    if space_key in keys and keys[space_key] & self.pbc.KEY_WAS_TRIGGERED:
                        break


if __name__ == "__main__":
    import time
    from robolearn_envs.pybullet.common import Drill, Plane

    pb_env = BulletEnv(is_render=True)
    pb_env.reset_simulation()

    obj = Drill(init_pos=[0., 0., 1000.0])
    obj2 = Plane()

    obj2.reset()
    obj.reset()

    for tt in range(int(1/pb_env.dt)):
        print('[1] pbenv_step: %02d' % pb_env.env_time, pb_env.sim_time)
        pb_env.render()
        pb_env.sim_step()
        time.sleep(pb_env.dt)

    for tt in range(int(1/pb_env.dt)):
        print('[2] pbenv_step: %02d' % pb_env.env_time, pb_env.sim_time)
        pb_env.render()
        pb_env.sim_step()
        time.sleep(pb_env.dt)

    print("Correct!")
