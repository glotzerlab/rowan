"""Test the reflection of quaternions for various array sizes."""

import unittest

import numpy as np

import rowan

x = np.array([1, 0, 0])
y = np.array([0, 1, 0])
z = np.array([0, 0, 1])

x_quat = np.array([0, 1, 0, 0])
y_quat = np.array([0, 0, 1, 0])
z_quat = np.array([0, 0, 0, 1])


class TestReflect(unittest.TestCase):
    """Test the core reflection operation."""

    def from_mirror_plane_single(self):
        """Test simple reflection about axes."""
        x_plane = rowan.from_mirror_plane(x[0], x[1], x[2])
        y_plane = rowan.from_mirror_plane(y[0], y[1], y[2])
        z_plane = rowan.from_mirror_plane(z[0], z[1], z[2])
        assert np.all(x_plane == x_quat)
        assert np.all(y_plane == y_quat)
        assert np.all(z_plane == z_quat)

    def test_single_quaternion(self):
        """Testing trivial reflections about planes."""
        x_reflect = rowan.reflect(x_quat, x)
        y_reflect = rowan.reflect(y_quat, y)
        z_reflect = rowan.reflect(z_quat, z)

        assert np.all(x_reflect == -x)
        assert np.all(y_reflect == -y)
        assert np.all(z_reflect == -z)

    def test_broadcast(self):
        """Ensure broadcasting works."""
        x_plane = rowan.from_mirror_plane([x[0], x[0]], x[1], x[2])
        assert np.all(x_plane == x_quat[np.newaxis, :].repeat(2, axis=0))
        x_reflect = rowan.reflect(x_plane, x)
        assert np.all(x_reflect == -x[np.newaxis, :].repeat(2, axis=0))
