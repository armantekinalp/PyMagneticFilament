__all__ = ["EulerForward", "RungeKutta4"]
import numpy as np


class EulerForward:
    def __init__(self):
        pass

    def __call__(self, system, time: float, dt: float, *args, **kwargs):

        system.state[:] += dt * system(time)


class RungeKutta4:
    def __init__(self):
        pass

    def __call__(self, system, time: float, dt: float, *args, **kwargs):
        initial_state = system.state.copy()

        # First stage
        k1 = system(time)

        # Second stage
        system.state[:] = initial_state + dt * k1 / 2
        k2 = system(time + dt / 2)

        # Third stage
        system.state[:] = initial_state + dt * k2 / 2
        k3 = system(time + dt / 2)

        # Fourth stage
        system.state[:] = initial_state + dt * k3
        k4 = system(time + dt)

        # RK
        system.state[:] = initial_state + 1 / 6 * dt * (k1 + 2 * k2 + 2 * k3 + k4)
