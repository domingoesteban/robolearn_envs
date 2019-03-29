import numpy as np


class RobolearnState(np.ndarray):

    def __init__(self, values):
        self.values = np.array(values)

    def __len__(self):
        return len(self.values)

    # @property
    # def dim(self):
    #     return len(self.values)

    def __add__(self, other):
        return RobolearnState(np.concatenate((self.values, other.values)))

    def __radd__(self, other):
        return RobolearnState(np.concatenate((other.values, self.values)))

    def __repr__(self):
        return "RobolearnState: " + repr(self.values)

    def __call__(self, *args, **kwargs):
        return self.values
