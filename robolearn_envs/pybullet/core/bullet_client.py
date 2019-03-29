import functools
import inspect
import pybullet


class BulletClient(object):
    """A wrapper for pybullet to manage different clients."""

    def __init__(self, connection_mode=pybullet.DIRECT, options=""):
        """Create a simulation and connect to it."""
        self._client = pybullet.connect(pybullet.SHARED_MEMORY)
        self._connection_mode = connection_mode
        if self._client < 0:
            self._client = pybullet.connect(connection_mode, options=options)

    def disconnect(self):
        try:
            pybullet.disconnect(physicsClientId=self._client)
        except pybullet.error:
            pass

    # def __del__(self):
    #     """Clean up connection if not already done."""
    #     self.disconnect()

    def __getattr__(self, name):
        """Inject the client id into Bullet functions."""
        attribute = getattr(pybullet, name)
        if inspect.isbuiltin(attribute):
            attribute = functools.partial(attribute,
                                          physicsClientId=self._client)
        return attribute

    @property
    def is_gui(self):
        return self._connection_mode == pybullet.GUI

    @property
    def is_direct(self):
        return self._connection_mode == pybullet.DIRECT
