import numpy as np
import typing as T
from src.utils.math import signed_sin
from scipy.spatial.transform import Rotation as R
from src.env.control.base_control import BaseControl


class PoseControl(BaseControl):
    """
    This class handles the random generation of the heading vector 
    and the target point for the robot to follow.
    """

    def __init__(self):
        super().__init__()
        self._heading = np.array([0.0, 0.0])          # Unit heading vector (pointing along x-axis)

    # ------------------------------------------------------------------ #
    #                         PROPERTIES                                  #
    # ------------------------------------------------------------------ #

    @property
    def heading(self) -> np.ndarray:
        """Get the heading unit vector (2D)."""
        return self._heading

    @heading.setter
    def heading(self, value: np.ndarray) -> None:
        """Set the heading vector. It will be normalised automatically."""
        value = np.asarray(value, dtype=np.float64)
        norm = np.linalg.norm(value)
        if norm < 1e-8:
            raise ValueError("Heading vector cannot be zero.")
        self._heading = value / norm

    @property
    def heading_angle(self) -> float:
        """Get the heading yaw angle in radians (atan2 of y/x component)."""
        return float(np.arctan2(self._heading[1], self._heading[0]))

    @heading_angle.setter
    def heading_angle(self, versor: np.ndarray) -> None:
        """Set the heading from a yaw angle (radians). Z component is kept at 0."""
        self._heading = np.array([versor[0], versor[1]])

    # ------------------------------------------------------------------ #
    #                    RANDOM GENERATION METHODS                        #
    # ------------------------------------------------------------------ #

    def update(self):
        """
        Update the heading vector based on the current heading angle.
        """
        # Update the heading vector adding or subtracting a random angle between 10 and -10 degrees wrt the current angle
        delta_angle = np.deg2rad(np.random.uniform(-10, 10))
        current_angle = self.heading_angle
        new_angle = current_angle + delta_angle
        self.heading_angle = np.array([np.cos(new_angle), np.sin(new_angle)])

    def generate_random(self) -> np.ndarray:
        """
        Generate a random heading direction uniformly sampled on the unit circle (XY plane).
        
        Returns:
            np.ndarray: The new heading unit vector (2D).
        """
        yaw = np.random.uniform(-np.pi, np.pi)
        self._heading = np.array([np.cos(yaw), np.sin(yaw)])
        return self._heading.copy()

    # ------------------------------------------------------------------ #
    #                        UTILITY METHODS                              #
    # ------------------------------------------------------------------ #

    def error_with_quaternion(self, quat: np.ndarray) -> float:
        """
        Convert a quaternion to a heading unit vector in the XY plane and compute the signed angle error.

        Args:
            quat: A 4D array representing the quaternion (w, x, y, z).
        Returns:
            float: The signed angle error in radians.
        """
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]]) # Rearrange to [x, y, z, w]
        rot_matrix = r.as_matrix()
        # The forward direction in the robot's local frame is typically along the x-axis
        forward_vector = rot_matrix[:2, 0]  # Get the first column of the rotation
        # Normalize the forward vector
        norm = np.linalg.norm(forward_vector)
        if norm > 1e-6:
            forward_vector /= norm
        else:
            forward_vector = np.array([0.0, 0.0])
        
        return self.error(forward_vector)

    def error(
        self, 
        direction_vector
        ) -> float:
        """
        Compute the signed angle error between the current heading and a given direction vector.

        Args:
            direction_vector: A 2D vector representing the direction to compare with.
        Returns:
            float: Signed angle error in radians, positive if the direction is to the left of the heading.
        """
        dir_vec = np.asarray(direction_vector, dtype=np.float64).flatten()
        if dir_vec.shape[0] == 2:
            dir_vec = np.array([dir_vec[0], dir_vec[1]])
        else:
            raise ValueError(f"Direction vector must have 2 elements, got {dir_vec.shape[0]}.")
        return float(signed_sin(self._heading[:2], dir_vec[:2]))

    def reset(self) -> None:
        """Reset all pose control parameters to their default values."""
        self._heading = np.array([1.0, 0.0])

    # ------------------------------------------------------------------ #
    #                         DUNDER METHODS                             #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (f"PoseControl(heading={self._heading}, angle={np.degrees(self.heading_angle):.1f}°)")