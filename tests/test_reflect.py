"""Test the reflection of quaternions for various array sizes"""
from __future__ import division, print_function, absolute_import

import unittest
import numpy as np

import hamilton as quaternion

x = np.array([1, 0, 0])
y = np.array([0, 1, 0])
z = np.array([0, 0, 1])

x_quat = np.array([0, 1, 0, 0])
y_quat = np.array([0, 0, 1, 0])
z_quat = np.array([0, 0, 0, 1])


class TestReflect(unittest.TestCase):
    """Test the core reflection operation"""

    def from_mirror_plane_single(self):
        """Simple test of reflect about axes"""
        x_plane = quaternion.from_mirror_plane(x[0], x[1], x[2])
        y_plane = quaternion.from_mirror_plane(y[0], y[1], y[2])
        z_plane = quaternion.from_mirror_plane(z[0], z[1], z[2])
        self.assertTrue(np.all(x_plane == x_quat))
        self.assertTrue(np.all(y_plane == y_quat))
        self.assertTrue(np.all(z_plane == z_quat))

    def test_single_quaternion(self):
        """Testing trivial reflections about planes"""
        x_reflect = quaternion.reflect(x_quat, x)
        y_reflect = quaternion.reflect(y_quat, y)
        z_reflect = quaternion.reflect(z_quat, z)

        self.assertTrue(np.all(x_reflect == -x))
        self.assertTrue(np.all(y_reflect == -y))
        self.assertTrue(np.all(z_reflect == -z))

    def test_broadcast(self):
        """Ensure broadcasting works"""
        x_plane = quaternion.from_mirror_plane([x[0], x[0]], x[1], x[2])
        self.assertTrue(np.all(x_plane == x_quat[np.newaxis, :].repeat(
            2, axis=0)))
        x_reflect = quaternion.reflect(x_plane, x)
        self.assertTrue(np.all(x_reflect == -x[np.newaxis, :].repeat(
            2, axis=0)))
