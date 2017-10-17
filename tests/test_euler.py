"""Test converting quaternions to and from Euler angles"""

import unittest
import numpy as np

import quaternion as quaternion

zero = np.array([0, 0, 0, 0])
one = np.array([1, 0, 0, 0])


class TestEuler(unittest.TestCase):
    """Test Euler angle conversions"""

    def test_from_euler(self):
        angles = np.array([0, 0, 0])

        self.assertTrue(np.all(
            quaternion.from_euler(angles) ==
            np.array([1, 0, 0, 0])
        ))

        angles = np.array([np.pi / 2, np.pi / 2, 0])
        self.assertTrue(np.allclose(
            quaternion.from_euler(angles),
            np.array([0.5, 0.5, 0.5, -0.5])
        ))

    def test_to_euler(self):
        v = one
        self.assertTrue(np.all(
            quaternion.to_euler(v) == np.array([0, 0, 0])
        ))

        v = np.array([0.5, 0.5, 0.5, 0.5])
        self.assertTrue(np.all(
            quaternion.to_euler(v) == np.array([np.pi / 2, 0, np.pi / 2])
        ))
