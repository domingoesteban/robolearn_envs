import numpy as np
import pybullet as pb
from robolearn_envs.pybullet.core.bullet_colors import pb_colors


class BulletMultibody(object):
    def __init__(self, pybullet_client=None):

        self._links = None  # All available links
        self._joints = None  # All available joints
        self._ordered_joints = None  # User-specified joints

        # This should be filled later
        self.id = None  # Bullet unique ID
        self.name = None
        self._bodies_uids = None  # Some xml files have several bodies

        self.pybullet_client = pybullet_client

    @property
    def pybullet_client(self):
        if self._pybullet_client is None:
            raise ValueError("No pybullet client has been assigned to this "
                             "%s instance!" % self.__class__.__name__.upper())
        return self._pybullet_client

    @pybullet_client.setter
    def pybullet_client(self, pb_client):
        self._pybullet_client = pb_client

        if self._links:
            for link in self._links.values():
                link.pbc = self._pybullet_client

        if self._joints:
            for joint in self._joints.values():
                joint.pbc = self._pybullet_client

    pbc = pybullet_client

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, bullet_id):
        self._id = bullet_id

    @property
    def name(self):
        if self._name is None:
            return self.pbc.getBodyInfo(self.id)
        return self._name

    @name.setter
    def name(self, name):
        if name is None:
            name = self.__class__.__name__.lower()
        self._name = name

    # ################# #
    # Base body methods #
    # ################# #
    def get_pose(self):
        return self.get_base_pose()

    def set_pose(self, position, orientation=(0, 0, 0, 1)):
        self.reset_base_pose(position, orientation)

    def set_position(self, position):
        self.set_base_position(position)

    def set_orientation(self, orientation):
        self.set_base_orientation(orientation)

    def reset_base_pose(self, position, orientation=(0, 0, 0, 1)):
        self.pbc.resetBasePositionAndOrientation(
            self.id, position, orientation
        )

    def set_base_pose(self, position, orientation=(0, 0, 0, 1)):
        self.reset_base_pose(position, orientation)

    def set_base_position(self, position):
        self.pbc.resetBasePositionAndOrientation(
            self.id, position, self.get_base_orientation()
        )

    def set_base_orientation(self, orientation):
        self.pbc.resetBasePositionAndOrientation(
            self.id, self.get_base_position(), orientation
        )

    def get_base_pose(self):
        return np.concatenate(
            self.pbc.getBasePositionAndOrientation(self.id), axis=0
        )

    def get_base_position(self):
        return np.array(
            self.pbc.getBasePositionAndOrientation(self.id)[0]
        )

    def get_base_orientation(self):
        return np.array(
            self.pbc.getBasePositionAndOrientation(self.id)[1]
        )

    def get_base_velocity(self):
        return np.concatenate(
            self.pbc.getBaseVelocity(self.id), axis=0
        )

    def get_base_linear_velocity(self):
        return np.array(self.pbc.getBaseVelocity(self.id)[0])

    def get_base_angular_velocity(self):
        return np.array(self.pbc.getBaseVelocity(self.id)[1])

    # ###### #
    # Bodies #
    # ###### #

    def get_total_bodies(self):
        return self.pbc.getNumJoints(self.id)+1

    def get_link_ids(self, link_names):

        def get_index(name):
            if name in self._links.keys():
                return self._links[name].id
            else:
                raise ValueError("Not available link")

        # list of links
        return [get_index(name) for name in link_names]

    # ######## #
    # Dynamics #
    # ######## #

    def change_dynamics(self, **kwargs):
        self.pbc.changeDynamics(
            bodyUniqueId=self.id,
            **kwargs
        )

    # ############# #
    # Miscellaneous #
    # ############# #
    def set_color(self, color):
        """Change to one color all the bodies of the robot.

        Args:
            color (str or tuple): color

        Returns:
            None

        """
        if issubclass(type(color), str):
            if color.lower() in pb_colors:
                color = pb_colors[color]
            else:
                raise ValueError("Color %s is not available. Choose one of "
                                 "the fowllowing: %s"
                                 % (color, [cc for cc in pb_colors.keys()]))
        color_list = [color for _ in range(self.get_total_bodies())]
        self.set_body_colors(color_list)

    def set_body_colors(self, colors):
        """Change the color in bodies according to a list of colors.

        Args:
            colors (list):

        Returns:
            None
        """
        color_count = 0
        for uid in self._bodies_uids:
            if self.pbc.getNumJoints(uid) == 0:
                self.pbc.changeVisualShape(uid, -1,
                                           rgbaColor=colors[color_count])
                color_count += 1

            for joint in range(self.pbc.getNumJoints(uid)):
                self.pbc.changeVisualShape(uid, joint,
                                           rgbaColor=colors[color_count])
                color_count += 1
