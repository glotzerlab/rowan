"""Test converting quaternions to and from Euler angles"""

import unittest
import numpy as np
import os

import hamilton as quaternion

zero = np.array([0, 0, 0, 0])
one = np.array([1, 0, 0, 0])

TESTDATA_FILENAME = os.path.join(
    os.path.dirname(__file__),
    'files/test_arrays.npz')
with np.load(TESTDATA_FILENAME) as data:
    euler_angles = data['euler_angles']
    euler_quaternions = data['euler_quats']

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
            quaternion.from_euler(angles, 'zyx','intrinsic'),
            np.array([0.5, -0.5, 0.5, 0.5])
        ))

        # More complicated test, checks 2d arrays
        # and more complex angles
        self.assertTrue(
                np.allclose(
                    quaternion.from_euler(euler_angles, 'zyz','intrinsic'),
                    euler_quaternions
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
