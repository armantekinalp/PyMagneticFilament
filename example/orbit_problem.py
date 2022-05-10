import numpy as np
from magnetic_filament.integrator import integrate
from magnetic_filament.time_stepper import *
from magnetic_filament.system import BaseSystem
import matplotlib

matplotlib.use("Agg")  # Must be before importing matplotlib.pyplot or pylab!
from matplotlib import pyplot as plt
from collections import defaultdict


class Orbit(BaseSystem):
    def __init__(self, initial_conditions, post_processing_dict, step_skip: int):

        self.state = np.zeros((2), dtype=np.float64)
        self.state[:] = initial_conditions

        # Create references for later data collection
        self.x = np.ndarray.view(self.state[0:1])
        self.y = np.ndarray.view(self.state[1:])

        self.state_dot = np.zeros(self.state.shape, dtype=np.float64)

        # Data recording
        self.post_processing_dict = post_processing_dict
        self.step_skip = step_skip

    def __call__(self, time, *args):

        self.state_dot[0] = -self.y

        self.state_dot[1] = self.x

        return self.state_dot

    def apply_forces_or_torques(self, time):
        pass

    def compute_dissipation(self, time):
        pass

    def callback(self, time, current_step: int):
        if current_step % self.step_skip == 0:
            self.post_processing_dict["time"].append(time)
            self.post_processing_dict["step"].append(current_step)
            self.post_processing_dict["x"].append(self.x.copy())
            self.post_processing_dict["y"].append(self.y.copy())


period = 2 * np.pi
initial_conditions = np.array([1, 0])

time_step = 1e-5
final_time = period
total_steps = int(final_time / time_step)
rendering_fps = 30
step_skip = int(1.0 / (rendering_fps * time_step))

# Euler time-integrator
post_processing_dict_euler = defaultdict(list)
oscillator_euler = Orbit(initial_conditions, post_processing_dict_euler, step_skip)
integrate(oscillator_euler, EulerForward(), total_steps, time_step)

time_euler = np.array(post_processing_dict_euler["time"])
x_euler = np.array(post_processing_dict_euler["x"])
y_euler = np.array(post_processing_dict_euler["y"])

# RK4 time-integrator
post_processing_dict_rk4 = defaultdict(list)
oscillator_rk4 = Orbit(initial_conditions, post_processing_dict_rk4, step_skip)
integrate(oscillator_rk4, RungeKutta4(), total_steps, time_step)

time_rk4 = np.array(post_processing_dict_rk4["time"])
x_rk4 = np.array(post_processing_dict_rk4["x"])
y_rk4 = np.array(post_processing_dict_rk4["y"])


# analytical
x = np.cos(time_euler / period)
y = np.sin(time_euler / period)

# Plot the results
plt.rcParams.update({"font.size": 22})
fig = plt.figure(figsize=(10, 10), frameon=True, dpi=150)

axs = []
axs.append(plt.subplot2grid((1, 1), (0, 0)))

axs[0].plot(
    x,
    y,
    label="analytical",
)
axs[0].plot(
    x_euler,
    y_euler,
    label="euler",
)
axs[0].plot(
    x_rk4,
    y_rk4,
    "--",
    label="rk4",
)
axs[0].set_xlabel("x", fontsize=20)
axs[0].set_ylabel("y", fontsize=20)
plt.tight_layout()
fig.align_ylabels()
fig.legend(prop={"size": 20})
fig.savefig("orbit.png")
plt.close(plt.gcf())


# Error-plots
error_euler = np.abs(x - x_euler[:, 0])
error_rk4 = np.abs(x - x_rk4[:, 0])

plt.rcParams.update({"font.size": 22})
fig = plt.figure(figsize=(10, 10), frameon=True, dpi=150)

axs = []
axs.append(plt.subplot2grid((1, 1), (0, 0)))
axs[0].semilogy(
    time_euler,
    error_euler,
    "-",
    label="euler",
)
axs[0].semilogy(
    time_rk4,
    error_rk4,
    "--",
    label="rk4",
)
axs[0].set_xlabel("time", fontsize=20)
axs[0].set_ylabel("relative error", fontsize=20)
plt.tight_layout()
fig.align_ylabels()
fig.legend(prop={"size": 20})
fig.savefig("orbit_error.png")
plt.close(plt.gcf())
