import numpy as np


class BaseSystem:
    def __init__(self):
        # requires state variable
        pass

    def __call__(self, time, *args):
        self.apply_forces_or_torques(time)
        pass

    def apply_forces_or_torques(self, time: np.float64):
        pass

    def compute_dissipation(self, time: np.float64):
        pass

    def callback(self, time, current_step: int):
        pass
