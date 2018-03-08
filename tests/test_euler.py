"""Test converting quaternions to and from Euler angles"""
from __future__ import division, print_function, absolute_import

import numpy as np
import os

import hamilton as quaternion
import unittest

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
            quaternion.from_euler(angles, 'zyx', 'intrinsic'),
            np.array([0.5, -0.5, 0.5, 0.5])
        ))

        # More complicated test, checks 2d arrays
        # and more complex angles
        self.assertTrue(
            np.allclose(
                quaternion.from_euler(euler_angles, 'zyz', 'intrinsic'),
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

        # More complicated test, checks 2d arrays
        # and more complex angles
        self.assertTrue(
            np.allclose(
                quaternion.to_euler(euler_quaternions, 'zyz', 'intrinsic'),
                euler_angles
            ))

    def test_from_to_euler(self):
        np.random.seed(0)
        quats = quaternion.normalize(np.random.rand(25, 4))
        conventions = ['xzx', 'xyx', 'yxy', 'yzy', 'zyz', 'zxz',
                       'xzy', 'xyz', 'yxz', 'yzx', 'zyx', 'zxy']
        axis_types = ['extrinsic', 'intrinsic']

        for convention in conventions:
            for axis_type in axis_types:
                out = quaternion.from_euler(
                    quaternion.to_euler(quats, convention, axis_type),
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
        np.random.seed(0)
        angles_euler = np.pi*np.random.rand(100, 3)
        conventions_euler = ['xzx', 'xyx', 'yxy', 'yzy', 'zyz', 'zxz']

        angles_tb = np.pi*np.random.rand(100, 3)
        # For Tait-Bryan angles the second angle must be between -pi/2 and pi/2
        angles_tb[:, 1] -= np.pi/2
        conventions_tb = ['xzy', 'xyz', 'yxz', 'yzx', 'zyx', 'zxy']

        axis_types = ['extrinsic', 'intrinsic']

        for convention in conventions_euler:
            for axis_type in axis_types:
                out = quaternion.to_euler(
                    quaternion.from_euler(angles_euler, convention, axis_type),
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
                out = quaternion.to_euler(
                    quaternion.from_euler(angles_tb, convention, axis_type),
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
