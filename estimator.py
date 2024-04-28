from const import EstimatorConstant
import numpy as np
from typing import Tuple


class EKF:
    """
    Extended Kalman Filter class

    Args:
        estimator_constant : EstimatorConstant
            Constants known to the estimator.
    """

    def __init__(
            self,
            estimator_constant: EstimatorConstant,
    ):
        self.constant = estimator_constant

    def initialize(
            self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize the estimator with the mean and covariance of the initial
        estimate.

        Returns:
            xm : np.ndarray, dim: (num_states,)
                The mean of the initial state estimate. The order of states is
                given by x = [p_x, p_y, psi, tau, l].
            Pm : np.ndarray, dim: (num_states, num_states)
                The covariance of the initial state estimate. The order of
                states is given by x = [p_x, p_y, psi, tau, l].
        """
        xm = None
        Pm = None

        return xm, Pm

    def estimate(
            self,
            xm_prev: np.ndarray,
            Pm_prev: np.ndarray,
            inputs: np.ndarray,
            measurement: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate the state of the vehicle.

        Args:
            xm_prev : np.ndarray, dim: (num_states,)
                The mean of the previous posterior state estimate xm(k-1). The
                order of states is given by x = [p_x, p_y, psi, tau, l].
            Pm_prev : np.ndarray, dim: (num_states, num_states)
                The covariance of the previous posterior state estimate Pm(k-1).
                The order of states is given by x = [p_x, p_y, psi, tau, l].
            inputs : np.ndarray, dim: (num_inputs,)
                System inputs from time step k-1, u(k-1). The order of the
                inputs is given by u = [u_delta, u_c].
            measurement : np.ndarray, dim: (num_measurement,)
                Sensor measurements from time step k, z(k). The order of the
                measurements is given by z = [z_px, z_py, z_psi, z_tau].

        Returns:
            xm : np.ndarray, dim: (num_states,)
                The mean of the posterior estimate xm(k). The order of states is
                given by x = [p_x, p_y, psi, tau, l].
            Pm : np.ndarray, dim: (num_states, num_states)
                The covariance of the posterior estimate Pm(k). The order of
                states is given by x = [p_x, p_y, psi, tau, l].
        """
        xm = None
        Pm = None

        return xm, Pm


class PF:
    """
    Particle Filter class

    Args:
        estimator_constant : EstimatorConstant
            Constants known to the estimator.
        noise : str
            Type of noise, either "Gaussian" or "Non-Gaussian".
    """
    def __init__(
            self,
            estimator_constant: EstimatorConstant,
            noise: str,
    ):
        self.constant = estimator_constant
        self.num_particles = 200  # you should fine tune this parameter
        if noise == "Gaussian" or noise == "Non-Gaussian":
            self.noise = noise
        else:
            raise ValueError(
                "Noise type not supported, should be either Gaussian or "
                "Non-Gaussian!"
            )

    def initialize(self) -> np.ndarray:
        """
        Initialize the estimator with the particles.

        Returns:
            particles: np.ndarray, dim: (num_states, num_particles)
                The particles corresponding to the initial state estimate. The
                order of states is given by x = [p_x, p_y, psi, tau, l].
        """
        particles = None

        return particles

    def estimate(
            self,
            particles: np.ndarray,
            inputs: np.ndarray,
            measurement: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate the state of the vehicle.

        Args:
            particles : np.ndarray, dim: (num_states, num_particles)
                The posteriors of the particles of the previous time step k-1.
                The order of states is given by x = [p_x, p_y, psi, tau, l].
            inputs : np.ndarray, dim: (num_inputs,)
                System inputs from time step k-1, u(k-1). The order of the
                inputs is given by u = [u_delta, u_c].
            measurement : np.ndarray, dim: (num_measurement,)
                Sensor measurements from time step k, z(k). The order of the
                measurements is given by z = [z_px, z_py, z_psi, z_tau].

        Returns:
            posteriors : np.ndarray, dim: (num_states, num_particles)
                The posterior particles at time step k. The order of states is
                given by x = [p_x, p_y, psi, tau, l].
        """
        posteriors = None
        
        if self.noise == "Non-Gaussian":
            # sample noises from the non-Gaussian distribution
            pass
        else:
            # sample noises from the Gaussian distribution
            pass

        return posteriors
