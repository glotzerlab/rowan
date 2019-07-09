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

        with self.assertRaises(ValueError):
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

    def test_zero_beta(self):
        """Check cases where beta is 0."""
        # The simplest failure cases will just involve two equal nonzero
        # quaternion components, which when normalized will be sqrt(2). These
        # simple cases correspond to alpha=pi/2 and gamma=0 (and beta=0).
        root2 = np.sqrt(2)/2

        # These are the quaternions that will fail because the normally used
        # entries in the matrix are 0, so the arctan functions return
        # meaningless values.
        test_quats = [
            ('xzx', 'intrinsic', (root2, root2, 0, 0)),
            ('xyx', 'intrinsic', (root2, root2, 0, 0)),
            ('xyx', 'intrinsic', (0, 0, -root2, -root2)),
            ('yxy', 'intrinsic', (root2, 0, root2, 0)),
            ('yzy', 'intrinsic', (root2, 0, root2, 0)),
            ('zyz', 'intrinsic', (root2, 0, 0, root2)),
            ('zxz', 'intrinsic', (root2, 0, 0, root2)),
            ('xzy', 'intrinsic', (root2, 0, 0, root2)),
            ('xzy', 'intrinsic', (-0.5, -0.5,  0.5, -0.5)),
            ('xyz', 'intrinsic', (root2, 0, root2, 0)),
            ('xyz', 'intrinsic', (-0.5, -0.5, -0.5, -0.5)),
            ('yxz', 'intrinsic', (root2, root2, 0, 0)),
            ('yxz', 'intrinsic', (0.5, 0.5, 0.5, -0.5)),
            ('yzx', 'intrinsic', (root2, 0, 0, root2)),
            ('yzx', 'intrinsic', (-0.5, -0.5, -0.5, -0.5)),
            ('zyx', 'intrinsic', (root2, 0, root2, 0)),
            ('zyx', 'intrinsic', (-0.5, 0.5, -0.5, -0.5)),
            ('zxy', 'intrinsic', (root2, root2, 0, 0)),
            ('zxy', 'intrinsic', (-0.5, -0.5, -0.5, -0.5)),
        ]

        # Choose simplest vector with all 3 components (otherwise tests won't
        # catch the problem because there's no component to rotate).
        test_vector = [1, 1, 1]
        for convention, axis_type, quat in test_quats:
            # Since angle representations may not be unique, checking that
            # quaternions are equal may not work. Instead we need to perform
            # multiple conversions to get back to quaternions and check that
            # the rotations are identical.
            # print("Running")
            euler = rowan.to_euler(
                    quat,
                    convention, axis_type
                    )
            converted = rowan.from_euler(
                *euler,
                convention, axis_type
            )
            try:
                self.assertTrue(
                    np.allclose(
                        rowan.rotate(quat, test_vector),
                        rowan.rotate(converted, test_vector),
                        atol=1e-6
                    ),
                    msg="\nFailed for convention {},\naxis type {},\nquat {},\nconverted = {}\nrotate1 = {}\nrotate2 = {}".format(
                        convention, axis_type, quat, converted,
                        rowan.rotate(quat, test_vector),
                        rowan.rotate(converted, test_vector)))
            except AssertionError as e:
                print(e)
