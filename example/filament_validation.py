import numpy as np
from magnetic_filament.integrator import integrate
from magnetic_filament.time_stepper import *
import matplotlib

matplotlib.use("Agg")  # Must be before importing matplotlib.pyplot or pylab!
from matplotlib import pyplot as plt
from collections import defaultdict
from magnetic_filament.filament import Filament


class FilamentSystems(Filament):
    def __init__(
        self,
        position,
        velocity,
        length,
        point_force,
        ramp_interval,
        post_processing_dict,
        step_skip,
        line_density,
        bending_stiffness,
        axial_stiffness,
        internal_damping_coeff,
        real_dtype=np.float64,
    ):
        super(FilamentSystems, self).__init__(
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
        self.point_force = point_force
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
        self.external_force[..., -1] = self.point_force * factor

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

            self.post_processing_dict["time"].append(time)
            self.post_processing_dict["step"].append(current_step)
            self.post_processing_dict["position"].append(self.position.copy())
            self.post_processing_dict["velocity"].append(self.velocity.copy())
            self.post_processing_dict["acceleration"].append(self.acceleration.copy())


n_elem = 50  # 400
base_length = 0.8  # 3.0
base_radius = 0.25
base_area = np.pi * base_radius ** 2
I = np.pi / 4 * base_radius ** 4
density = 5000
nu = 0.1
E = 1e6
EI = 0.01

# FilamentPosition
initial_position = np.zeros((2, n_elem))
initial_position[0] = np.linspace(0, base_length, n_elem)

initial_velocity = np.zeros((2, n_elem))

# External forces
point_force = np.array([0, -1e-3])  # np.array([0, -15.])
ramp_interval = 10.0

# Simulation parameters
final_time = 6.4  # 50

time_step = 1e-6
total_steps = int(final_time / time_step)
rendering_fps = 30
step_skip = int(1.0 / (rendering_fps * time_step))

# Euler integration
post_processing_dict_euler = defaultdict(list)
filament_object_euler = FilamentSystems(
    initial_position,
    initial_velocity,
    base_length,
    point_force,
    ramp_interval,
    post_processing_dict_euler,
    step_skip,
    line_density=1.2 / base_length,  # density*base_area,
    bending_stiffness=EI,  # E*I,
    axial_stiffness=1e3,  # E*base_area,
    internal_damping_coeff=0.8,  # 4.0,
    real_dtype=np.float64,
)
integrate(filament_object_euler, EulerForward(), total_steps, time_step)
position_euler = filament_object_euler.position

# # RK4 integration
time_step = 2e-4  # 1E-4/2/4#1E-4
final_time = 50
total_steps = int(final_time / time_step)
rendering_fps = 30
step_skip = int(1.0 / (rendering_fps * time_step))
post_processing_dict_rk4 = defaultdict(list)
filament_object_rk4 = FilamentSystems(
    initial_position,
    initial_velocity,
    base_length,
    point_force,
    ramp_interval,
    post_processing_dict_rk4,
    step_skip,
    line_density=1.2 / base_length,  # density*base_area,
    bending_stiffness=EI,  # E*I,
    axial_stiffness=1e3,  # E*base_area,
    internal_damping_coeff=0.8,  # 4.0,
    real_dtype=np.float64,
)


integrate(filament_object_rk4, RungeKutta4(), total_steps, time_step)
position_rk4 = filament_object_rk4.position

# Analytical solution
s = np.linspace(0, base_length, n_elem)
analytical_solution = (
    -np.linalg.norm(point_force) * base_length / (2 * EI) * s ** 2
    + np.linalg.norm(point_force) / (6 * EI) * s ** 3
)


# Plot the results
plt.rcParams.update({"font.size": 22})
fig = plt.figure(figsize=(10, 10), frameon=True, dpi=150)

axs = []
axs.append(plt.subplot2grid((1, 1), (0, 0)))

axs[0].plot(
    s,
    analytical_solution,
    label="analytical",
)
axs[0].plot(
    position_euler[0, :],
    position_euler[1, :],
    label="euler",
)
axs[0].plot(
    position_rk4[0, :],
    position_rk4[1, :],
    "--",
    label="rk4",
)

axs[0].set_xlabel("x", fontsize=20)
axs[0].set_ylabel("y", fontsize=20)
plt.tight_layout()
fig.align_ylabels()
fig.legend(prop={"size": 20})
fig.savefig("beam_deflection.png")
plt.close(plt.gcf())

np.linalg.norm(filament_object_euler.velocity[..., -1])
np.linalg.norm(filament_object_rk4.velocity[..., -1])
