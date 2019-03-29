class RoboLearnActuator(object):
    _state = None  # Internal state of the actuator
    _idx_list = []  # Indexes in action vector
    _limits = []  # Action limits for the actuators

    def actuate(self, action):
        NotImplementedError

    @property
    def dimension(self):
        return len(self._state) if self._state is not None else 0

    @property
    def dim(self):
        return self.dimension

    def set_indeces(self, indeces):
        if len(indeces) != self.dimension:
            raise ValueError("Number of indeces do not match with "
                             "actuator dimension.")

        self._idx_list = indeces

    @property
    def idx(self):
        return self._idx_list

    @property
    def limits(self):
        return self._limits


