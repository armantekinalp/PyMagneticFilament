import numpy as np
from numba import njit


class Filament:
    """
    Filament class.
    Attributes
    ----------
    position: numpy.ndarray
        2D (dim, N_nodes) array containing data with 'float' type.
        Array containing node position vectors.
    velocity: numpy.ndarray
        2D (dim, N_nodes) array containing data with 'float' type.
        Array containing node velocity vectors.
    length: float
        The length of filament with 'float' type.
    line_density: float
        The mass per unit length of filament with 'float' type.
    bending_stiffness: float
        The bending stiffness K_b of filament with 'float' type.
    axial_stiffness: float
        The axial stiffness K_s of filament with 'float' type.
    real_dtype: data type
        Data type for typecasting real values
    """

    def __init__(
        self,
        position,
        velocity,
        length,
        line_density=1.0,
        bending_stiffness=1.0,
        axial_stiffness=1e3,
        internal_damping_coeff=0.0,
        real_dtype=np.float64,
    ):

        self.position = position.astype(real_dtype)
        self.velocity = velocity.astype(real_dtype)
        self.dim = position.shape[0]
        assert (
            self.dim == 2 or self.dim == 3
        ), "Invalid problem dimension (only 2D and 3D)"
        self.N_nodes = position.shape[-1]
        self.length = real_dtype(length)
        self.ds = self.length / (self.N_nodes - 1)
        self.line_density = real_dtype(line_density)

        self.mass = (
            self.line_density * self.ds * np.ones(self.N_nodes).astype(real_dtype)
        )
        self.bending_stiffness = real_dtype(bending_stiffness)
        self.axial_stiffness = real_dtype(axial_stiffness)
        self.internal_damping_coeff = real_dtype(internal_damping_coeff)

        # Initialize the slope values
        self.slope = np.zeros((self.dim, self.N_nodes - 1)).astype(real_dtype)
        self.compute_slope(slope=self.slope, position=self.position, ds=self.ds)
        # Initialize the curvature values
        self.curvature = np.zeros_like(self.position)
        self.compute_curvature(curvature=self.curvature, slope=self.slope, ds=self.ds)

        # Initialize the tension values
        self.tension = np.zeros(self.N_nodes - 1).astype(real_dtype)
        self.compute_tension(self.tension, self.slope, self.axial_stiffness, real_dtype)
        self.external_force = np.zeros_like(self.position)
        self.total_force = np.zeros_like(self.position)
        self.acceleration = np.zeros_like(self.position)

        self.real_dtype = real_dtype

    @staticmethod
    @njit(cache=True, fastmath=True)
    def compute_slope(slope, position, ds):
        """
        Compute slopes
        """
        slope[...] = (position[..., 1:] - position[..., :-1]) / ds

    @staticmethod
    @njit(cache=True, fastmath=True)
    def compute_curvature(curvature, slope, ds):
        """
        Compute curvature
        """
        curvature[..., 1:-1] = (slope[..., 1:] - slope[..., :-1]) / ds

    @staticmethod
    @njit(cache=True, fastmath=True)
    def compute_tension(tension, slope, axial_stiffness, real_dtype):
        """
        Compute tension
        """
        tension[...] = axial_stiffness * (
            np.sqrt(np.sum(slope ** 2, axis=0)) - real_dtype(1)
        )

    @staticmethod
    @njit(cache=True, fastmath=True)
    def add_axial_force(force, tension, slope, ds, real_dtype):
        """
        Compute axial force from slope and tension and add to `force`
        """
        force[:, 1:-1] += tension[1:] * slope[:, 1:] - tension[:-1] * slope[:, :-1]
        force[:, 0] += tension[0] * slope[:, 0] / real_dtype(0.5)
        force[:, -1] += -tension[-1] * slope[:, -1] / real_dtype(0.5)

    @staticmethod
    @njit(cache=True, fastmath=True)
    def add_bending_force(force, curvature, ds, bending_stiffness, real_dtype):
        """
        Compute bending force from curvature and add to `force`
        """
        force[:, 1:-1] -= (
            bending_stiffness
            * (
                curvature[:, 2:]
                + curvature[:, :-2]
                - real_dtype(2) * curvature[:, 1:-1]
            )
            / ds
        )
        force[:, 0] -= bending_stiffness / ds * (curvature[:, 1] - curvature[:, 0])
        force[:, -1] += bending_stiffness / ds * (curvature[:, -1] - curvature[:, -2])

    @staticmethod
    @njit(cache=True, fastmath=True)
    def add_internal_damping_force(force, velocity, internal_damping_coeff, ds):
        """
        Add internal damping force
        """
        force[...] = force - internal_damping_coeff * velocity * ds

    @staticmethod
    @njit(cache=True, fastmath=True)
    def add_external_force(force, external_force):
        """
        Add external force `external_force` to `force`
        """
        force[...] = force + external_force

    def compute_acceleration(self):
        """
        Compute acceleration
        """
        self.compute_slope(self.slope, self.position, self.ds)

        # Filament has free end by default
        self.compute_curvature(self.curvature, self.slope, self.ds)

        self.compute_tension(
            self.tension, self.slope, self.axial_stiffness, self.real_dtype
        )

        self.total_force[...] = self.real_dtype(0.0)
        self.add_axial_force(
            self.total_force, self.tension, self.slope, self.ds, self.real_dtype
        )
        self.add_bending_force(
            self.total_force,
            self.curvature,
            self.ds,
            self.bending_stiffness,
            self.real_dtype,
        )
        self.add_external_force(self.total_force, self.external_force)
        self.add_internal_damping_force(
            self.total_force,
            self.velocity,
            self.internal_damping_coeff,
            self.ds,
        )

        self.acceleration[...] = self.total_force / self.mass
