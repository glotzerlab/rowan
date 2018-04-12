"""Test converting quaternions to and from Euler angles"""
from __future__ import division, print_function, absolute_import

import unittest
import numpy as np
import os

import rowan

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
        """Convert Euler angles to quaternions"""
        alpha, beta, gamma = [0, 0, 0]
        self.assertTrue(np.all(
            rowan.from_euler(alpha, beta, gamma) ==
            np.array([1, 0, 0, 0])
        ))

        alpha, beta, gamma = [np.pi / 2, np.pi / 2, 0]
        self.assertTrue(np.allclose(
            rowan.from_euler(alpha, beta, gamma,
                             'zyx', 'intrinsic'),
            np.array([0.5, -0.5, 0.5, 0.5])
        ))

        # Confirm broadcasting works from different Euler angles
        alpha, beta, gamma = [[0, np.pi / 2], [0, np.pi / 2], 0]
        self.assertTrue(np.allclose(
            rowan.from_euler(alpha, beta, gamma),
            np.array([[1, 0, 0, 0], [0.5, -0.5, 0.5, 0.5]])
        ))

        alpha = [[0, np.pi / 2], [0, np.pi / 2]]
        beta = [0, np.pi / 2]
        gamma = 0
        self.assertTrue(np.allclose(
            rowan.from_euler(alpha, beta, gamma),
            np.array([[[1, 0, 0, 0], [0.5, -0.5, 0.5, 0.5]],
                     [[1, 0, 0, 0], [0.5, -0.5, 0.5, 0.5]]])
        ))

        # More complicated test, checks 2d arrays and more complex angles
        alpha = euler_angles[:, 0]
        beta = euler_angles[:, 1]
        gamma = euler_angles[:, 2]
        self.assertTrue(
            np.allclose(
                rowan.from_euler(alpha, beta, gamma, 'zyz', 'intrinsic'),
                euler_quaternions
            ))

        # Ensure proper errors are raised
        with self.assertRaises(ValueError):
            rowan.from_euler(alpha, beta, gamma, 'foo', 'intrinsic')

        with self.assertRaises(ValueError):
            rowan.from_euler(alpha, beta, gamma, 'foo', 'extrinsic')

        with self.assertRaises(ValueError):
            rowan.from_euler(alpha, beta, gamma, 'zyz', 'bar')

    def test_to_euler(self):
        """Test conversion to Euler angles"""
        v = one
        self.assertTrue(np.all(
            rowan.to_euler(v) == np.array([0, 0, 0])
        ))

        v = np.array([0.5, 0.5, 0.5, 0.5])
        self.assertTrue(np.all(
            rowan.to_euler(v) == np.array([np.pi / 2, 0, np.pi / 2])
        ))

        # More complicated test, checks 2d arrays
        # and more complex angles
        self.assertTrue(
            np.allclose(
                rowan.to_euler(euler_quaternions, 'zyz', 'intrinsic'),
                euler_angles
            ))

        # Ensure proper errors are raised
        with self.assertRaises(ValueError):
            rowan.to_euler(euler_quaternions, 'foo', 'intrinsic')

        with self.assertRaises(ValueError):
            rowan.to_euler(euler_quaternions, 'foo', 'extrinsic')

        with self.assertRaises(ValueError):
            rowan.to_euler(euler_quaternions, 'zyz', 'bar')

        with self.assertRaises(ValueError):
            rowan.to_euler(2*one)

        with self.assertRaises(ZeroDivisionError):
            rowan.to_euler(zero)

    def test_from_to_euler(self):
        """2-way conversion starting from Euler angles"""
        np.random.seed(0)
        quats = rowan.normalize(np.random.rand(25, 4))
        conventions = ['xzx', 'xyx', 'yxy', 'yzy', 'zyz', 'zxz',
                       'xzy', 'xyz', 'yxz', 'yzx', 'zyx', 'zxy']
        axis_types = ['extrinsic', 'intrinsic']

        for convention in conventions:
            for axis_type in axis_types:
                euler = rowan.to_euler(quats, convention, axis_type)
                out = rowan.from_euler(
                        euler[..., 0], euler[..., 1], euler[..., 2],
                        convention, axis_type
                )
                self.assertTrue(
                    np.all(
                        np.logical_or(
                            np.isclose(out - quats, 0),
                            np.isclose(out + quats, 0)
                        )
                    ),
                    msg="Failed for convention {}, axis type {}".format(
                        convention, axis_type))

    def test_to_from_euler(self):
        """2-way conversion starting from quaternions"""
        np.random.seed(0)
        angles_euler = np.pi*np.random.rand(100, 3)
        conventions_euler = ['xzx', 'xyx', 'yxy', 'yzy', 'zyz', 'zxz']

        # For Tait-Bryan angles the second angle must be between -pi/2 and pi/2
        angles_tb = angles_euler.copy()
        angles_tb[:, 1] -= np.pi/2
        conventions_tb = ['xzy', 'xyz', 'yxz', 'yzx', 'zyx', 'zxy']

        axis_types = ['extrinsic', 'intrinsic']

        for convention in conventions_euler:
            for axis_type in axis_types:
                out = rowan.to_euler(
                    rowan.from_euler(
                        angles_euler[..., 0], angles_euler[..., 1],
                        angles_euler[..., 2], convention, axis_type),
                    convention, axis_type
                )
                self.assertTrue(
                    np.all(
                        np.logical_or(
                            np.isclose(out - angles_euler, 0),
                            np.isclose(out + angles_euler, 0)
                        )
                    ),
                    msg="Failed for convention {}, axis type {}".format(
                        convention, axis_type))

        for convention in conventions_tb:
            for axis_type in axis_types:
                out = rowan.to_euler(
                    rowan.from_euler(
                        angles_tb[..., 0], angles_tb[..., 1],
                        angles_tb[..., 2], convention, axis_type),
                    convention, axis_type
                )
                self.assertTrue(
                    np.all(
                        np.logical_or(
                            np.isclose(out - angles_tb, 0),
                            np.isclose(out + angles_tb, 0)
                        )
                    ),
                    msg="Failed for convention {}, axis type {}".format(
                        convention, axis_type))
