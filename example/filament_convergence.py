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


n_elem_list = [20, 50, 100, 150, 200, 400]
time_step_list = [4e-4, 4e-4, 1e-4, 1e-4 / 2, 1e-4 / 2, 1e-4 / 8]
l1_error = []
l2_error = []
l_inf_error = []
error_list = []

for i in range(len(n_elem_list)):
    n_elem = n_elem_list[i]
    base_length = 0.8
    EI = 0.01

    # FilamentPosition
    initial_position = np.zeros((2, n_elem))
    initial_position[0] = np.linspace(0, base_length, n_elem)

    initial_velocity = np.zeros((2, n_elem))

    # External forces
    point_force = np.array([0, -1e-3])
    ramp_interval = 10.0

    # Simulation parameters
    final_time = 0.001  # 50

    # # RK4 integration
    time_step = time_step_list[i]  # 1E-4/8#1E-4
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
        line_density=1.2 / base_length,
        bending_stiffness=EI,
        axial_stiffness=1e3,
        internal_damping_coeff=0.8,
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

    error = (position_rk4[1, :] - analytical_solution) / base_length
    l1_error.append(np.linalg.norm(error, ord=1))
    l2_error.append(np.linalg.norm(error, ord=2))
    l_inf_error.append(np.linalg.norm(error, ord=np.inf))
    error_list.append(error)

n_elem_list = np.array(n_elem_list)
dx = base_length / n_elem_list

l1_error = np.array(l1_error * dx)
l2_error = np.array(l2_error * dx ** 0.5)
l_inf_error = np.array(l_inf_error)
error_list = np.array(error_list)


np.savez(
    "../filament_convergence.npz",
    l1_error=l1_error,
    l2_error=l2_error,
    l_inf_error=l_inf_error,
    error_list=error_list,
    n_elem_list=n_elem_list,
)

# my_data = np.load("filament_convergence.npz")
# l1_error = my_data["l1_error"]
# l2_error = my_data["l2_error"]
# l_inf_error = my_data["l_inf_error"]
# n_elem_list = my_data["n_elem_list"]

scale_second_order = dx ** 2 * l2_error[0] / dx[0] ** 2
scale_first_order = dx * l1_error[0] / dx[0]

# Plot the results
plt.rcParams.update({"font.size": 22})
fig = plt.figure(figsize=(10, 10), frameon=True, dpi=150)

axs = []
axs.append(plt.subplot2grid((1, 1), (0, 0)))
axs[0].grid(b=True, which="major", color="lightgray", linestyle="-")
axs[0].grid(b=True, which="minor", color="lightgray", linestyle="-")
axs[0].loglog(
    n_elem_list,
    l1_error,
    label=r"l$_{1}$",
)
axs[0].loglog(
    n_elem_list,
    l2_error,
    label=r"l$_{2}$",
)
axs[0].loglog(
    n_elem_list,
    l_inf_error,
    label=r"l$_{\infty}$",
)
axs[0].loglog(
    n_elem_list,
    scale_first_order,
    "--",
    c="b",
    label="O(1)",
)
axs[0].loglog(
    n_elem_list,
    scale_second_order,
    "--",
    c="r",
    label="O(2)",
)
axs[0].set_xlabel(r"$n_{elem}$", fontsize=30)
axs[0].set_ylabel(r"$|\epsilon$|", fontsize=30)


plt.tight_layout()
fig.align_ylabels()
fig.legend(prop={"size": 30})
fig.savefig("convergence_filament.png")
fig.savefig("convergence_filament.eps")
plt.close(plt.gcf())
