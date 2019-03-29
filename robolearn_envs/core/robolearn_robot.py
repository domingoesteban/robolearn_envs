import collections
import numpy as np


class RoboLearnRobot(object):
    _sensors = collections.OrderedDict()
    _actuators = collections.OrderedDict()
    _actuation_bounds = []
    _observation_bounds = []

    def reset_robot(self):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    # ####### #
    # Sensors #
    # ####### #

    def read_sensor(self):
        if self._sensors:
            for name, sensor in iter(self._sensors.items()):
                sensor.read()
        else:
            raise ValueError("No sensor has been configured for this robot.")

    @property
    def observation_bounds(self):
        return self._observation_bounds

    def update_observation_bounds(self):
        if self._sensors:
            self._observation_bounds = []
            for sensor in iter(self._sensors.values()):
                for limit in sensor.limits:
                    self._observation_bounds.append(limit)

    @property
    def sensor_dimension(self):
        return int(sum([sensor.dim
                        for sensor in self._sensors.values()])
                   )

    def add_sensor(self, sensor, name=None):
        if name is None:
            name = 'Sensor%02d' % len(self._sensors)
        self._sensors[name] = sensor

    def remove_sensor(self, name=None):
        if name is None:
            self._sensors = collections.OrderedDict()
        else:
            self._sensors.pop(name)

    def get_sensor(self, name):
        if name not in self._sensors.keys():
            raise ValueError("The robot does not have sensor %s."
                             % name)
        return self._sensors[name]

    # ######### #
    # Actuators #
    # ######### #

    def apply_action(self, action):
        assert (np.isfinite(action).all())
        if self._actuators:
            for name, actuator in iter(self._actuators.items()):
                actuator.actuate(action[actuator.idx])
        else:
            raise ValueError("No actuator has been configured for this robot.")

    @property
    def action_bounds(self):
        return self._actuation_bounds

    @property
    def low_action_bounds(self):
        return np.array([bound[0] for bound in self.action_bounds])

    @property
    def high_action_bounds(self):
        return np.array([bound[1] for bound in self.action_bounds])

    def update_actuation_bounds(self):
        if self._actuators:
            self._actuation_bounds = []
            for actuator in iter(self._actuators.values()):
                for limit in actuator.limits:
                    self._actuation_bounds.append(limit)

    @property
    def actuator_dimension(self):
        return int(np.sum([actuator.dim
                           for actuator in self._actuators.values()])
                   )

    def add_actuator(self, actuator, name=None):
        if name is None:
            name = 'Actuator%02d' % len(self._actuators)
        if self._actuators:
            new_idx = next(reversed(self._actuators.values())).idx[-1] + 1
        else:
            new_idx = 0
        actuator.set_indeces(np.arange(new_idx, new_idx + actuator.dim))

        self._actuators[name] = actuator

        self.update_actuation_bounds()

    def remove_actuator(self, name=None):
        if name is None:
            self._actuators = collections.OrderedDict()
        else:
            self._actuators.pop(name)

    def get_actuator(self, name):
        if name not in self._actuators.keys():
            raise ValueError("The robot does not have actuator %s."
                             % name)
        return self._actuators[name]

    @property
    def observation_dim(self):
        return self.sensor_dimension

    obs_dim = observation_dim

    @property
    def action_dim(self):
        return self.actuator_dimension

    act_dim = action_dim
