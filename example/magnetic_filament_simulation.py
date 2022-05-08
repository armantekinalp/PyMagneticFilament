import numpy as np
import matplotlib

matplotlib.use("Agg")  # Must be before importing matplotlib.pyplot or pylab!
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib import cm
from mpl_toolkits.mplot3d import proj3d, Axes3D
from tqdm import tqdm
from matplotlib.patches import Circle
from typing import Dict, Sequence

from magnetic_filament.filament import Filament
from magnetic_filament.time_stepper import RungeKutta4
from magnetic_filament.integrator import integrate
from collections import defaultdict
from example.post_processing import plot_video_with_surface


class MagneticFilamentSystems(Filament):
    def __init__(
        self,
        position,
        velocity,
        length,
        angular_frequency,
        MBA,
        ramp_interval,
        post_processing_dict,
        step_skip,
        line_density,
        bending_stiffness,
        axial_stiffness,
        internal_damping_coeff,
        real_dtype=np.float64,
    ):
        super(MagneticFilamentSystems, self).__init__(
            position,
            velocity,
            length,
            line_density,
            bending_stiffness,
            axial_stiffness,
            internal_damping_coeff,
            real_dtype,
        )
        # requires state variable

        self.state = np.zeros((4, self.N_nodes))
        self.state_dot = np.zeros((4, self.N_nodes))

        self.state[0, :] = self.position[0, :]
        self.state[1, :] = self.position[1, :]
        self.state[2, :] = self.velocity[0, :]
        self.state[3, :] = self.velocity[1, :]

        self.state_dot[0, :] = self.velocity[0, :]
        self.state_dot[1, :] = self.velocity[1, :]
        self.state_dot[2, :] = self.acceleration[0, :]
        self.state_dot[3, :] = self.acceleration[1, :]

        # Force Variables
        self.MBA = MBA
        self.angular_frequency = angular_frequency
        self.ramp_interval = ramp_interval

        # Boundary condiitons
        self.start_position = self.position[..., 0:2].copy()

        # Callback
        self.post_processing_dict = post_processing_dict
        self.step_skip = step_skip

    def __call__(self, time, *args):
        self.apply_boundary_conditions()
        self.position[0, :] = self.state[0, :]
        self.position[1, :] = self.state[1, :]
        self.velocity[0, :] = self.state[2, :]
        self.velocity[1, :] = self.state[3, :]

        self.compute_acceleration()

        self.state_dot[0, :] = self.velocity[0, :]
        self.state_dot[1, :] = self.velocity[1, :]
        self.state_dot[2, :] = self.acceleration[0, :]
        self.state_dot[3, :] = self.acceleration[1, :]

        return self.state_dot

    def apply_forces_or_torques(self, time: np.float64):
        factor = min(1.0, time / self.ramp_interval)
        # self.external_force[0,-1] = factor * self.MBA * np.sin(self.angular_frequency*time)
        self.external_force[1, -1] = (
            factor * self.MBA * np.sin(self.angular_frequency * time)
        )

    def apply_boundary_conditions(self):
        self.state[0, 0] = self.start_position[0, 0]
        self.state[1, 0] = self.start_position[1, 0]
        self.state[0, 1] = self.start_position[0, 1]
        self.state[1, 1] = self.start_position[1, 1]
        self.state[2, 0] = 0
        self.state[3, 0] = 0
        self.state[2, 1] = 0
        self.state[3, 1] = 0

    def callback(self, time, current_step: int):
        if current_step % self.step_skip == 0:

            position_collection = np.zeros((3, self.position.shape[-1]))
            position_collection[0, :] = self.position[0, :]
            position_collection[1, :] = self.position[1, :]
            radius = 0.15 * np.ones((self.position.shape[-1] - 1))
            self.post_processing_dict["time"].append(time)
            self.post_processing_dict["step"].append(current_step)
            # self.post_processing_dict["position"].append(self.position.copy())
            self.post_processing_dict["velocity"].append(self.velocity.copy())
            self.post_processing_dict["acceleration"].append(self.acceleration.copy())
            self.post_processing_dict["radius"].append(radius.copy())
            self.post_processing_dict["position"].append(position_collection.copy())


n_elem = 50

base_length = 1.5
base_radius = 0.15
base_area = np.pi * base_radius ** 2
density = 2.39e3  # kg/m3
nu = 50
E = 1.85e5  # Pa
I = np.pi / 4 * base_radius ** 4
EI = E * I
volume = base_area * base_length
line_density = density * base_area

# External forces
magnetic_field_strength = 80e-3  # 80mT
# MBAL2_EI is a non-dimensional number from Wang 2019
MBAL2_EI = (
    3.82e-5 * magnetic_field_strength * 4e-3 / (1.85e5 * np.pi / 4 * (0.4e-3) ** 4)
)  # Magnetization magnitude * B * Length/(EI)
magnetization_density = (
    MBAL2_EI * E * I / (volume * magnetic_field_strength * base_length)
)
MBA = magnetization_density * magnetic_field_strength * base_area

angular_frequency_in_deg = 40
angular_frequency = np.deg2rad(angular_frequency_in_deg)
ramp_interval = 1.0


# FilamentPosition
initial_position = np.zeros((2, n_elem))
initial_position[0] = np.linspace(0, base_length, n_elem)

initial_velocity = np.zeros((2, n_elem))


# Simulation parameters
num_cycles = 3.0
final_time = 90

# # RK4 integration
time_step = 5e-4
total_steps = int(final_time / time_step)
rendering_fps = 5
step_skip = int(1.0 / (rendering_fps * time_step))
post_processing_dict_rk4 = defaultdict(list)
filament_object_rk4 = MagneticFilamentSystems(
    initial_position,
    initial_velocity,
    base_length,
    angular_frequency,
    MBA,
    ramp_interval,
    post_processing_dict_rk4,
    step_skip,
    line_density=line_density,  # density*base_area,
    bending_stiffness=EI,  # E*I,
    axial_stiffness=EI * 1e2,  # E*base_area,
    internal_damping_coeff=0.8 * 40,  # 4.0,
    real_dtype=np.float64,
)


integrate(filament_object_rk4, RungeKutta4(), total_steps, time_step)
position_rk4 = filament_object_rk4.position


plot_video_with_surface(
    [post_processing_dict_rk4],
    fps=rendering_fps,
    step=1,
    x_limits=(-4, 4),
    y_limits=(-4, 4),
    z_limits=(-4, 4),
)


position_history = np.array(post_processing_dict_rk4["position"])
deflection_history = position_history[:, 1, -1]
time = np.array(post_processing_dict_rk4["time"])

# Plot the results
time_list = [50, 55, 60, 65, 70, 75, 80, 85, 90]
for idx in time_list:
    plt.rcParams.update({"font.size": 22})
    fig = plt.figure(figsize=(10, 10), frameon=True, dpi=150)

    axs = []
    axs.append(plt.subplot2grid((1, 1), (0, 0)))
    axs[0].plot(
        position_history[idx, 0, :],
        position_history[idx, 1, :],
        label="time" + str(np.round(time[idx], 2)),
    )

    axs[0].set_xlabel("x", fontsize=20)
    axs[0].set_ylabel("y", fontsize=20)
    axs[0].set_xlim(-0.2, base_length + 0.2)
    axs[0].set_ylim(-0.2 - base_length, base_length + 0.2)
    plt.tight_layout()
    fig.align_ylabels()
    fig.legend(prop={"size": 20})
    fig.savefig("time" + str(np.round(time[idx], 2)) + " .png")
    plt.close(plt.gcf())


plt.rcParams.update({"font.size": 22})
fig = plt.figure(figsize=(10, 10), frameon=True, dpi=150)

axs = []
axs.append(plt.subplot2grid((1, 1), (0, 0)))
axs[0].plot(
    time,
    deflection_history / base_length,
    label=r"$\delta / L$",
)
axs[0].plot(
    time,
    np.sin(time * angular_frequency),
    label=r"$B(t)/B_{0}$",
)

axs[0].set_xlabel("Time [s]", fontsize=20)
axs[0].set_ylabel("Amplitude", fontsize=20)
axs[0].set_title(
    "Angular frequency " + str(angular_frequency_in_deg) + " deg/s", fontsize=20
)
plt.tight_layout()
fig.align_ylabels()

fig.legend(prop={"size": 20})
fig.savefig("magnetic_beam_tip_deflection_vs_time.png")
plt.close(plt.gcf())
