"""Test converting quaternions to and from Euler angles"""

import numpy as np
import os

import hamilton as quaternion
import unittest

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
            quaternion.from_euler(angles, 'zyx','intrinsic'),
            np.array([0.5, -0.5, 0.5, 0.5])
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


    def test_from_to_euler(self):
        np.random.seed(0)
        quats = quaternion.normalize(np.random.rand(25, 4))
        angles = quaternion.normalize(np.random.rand(25, 3))
        conventions = ['xzx', 'xyx', 'yxy', 'yzy', 'zyz', 'zxz',
                'xzy', 'xyz', 'yxz', 'yzx', 'zyx', 'zxy']
        axis_types = ['extrinsic', 'intrinsic']

        # Test one direction
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
        quats = quaternion.normalize(np.random.rand(25, 4))
        angles = quaternion.normalize(np.random.rand(25, 3))
        conventions = ['xzx', 'xyx', 'yxy', 'yzy', 'zyz', 'zxz',
                'xzy', 'xyz', 'yxz', 'yzx', 'zyx', 'zxy']
        axis_types = ['extrinsic', 'intrinsic']

        for convention in conventions:
            for axis_type in axis_types:
                out = quaternion.to_euler(
                    quaternion.from_euler(angles, convention, axis_type),
                    convention, axis_type
                    )
                self.assertTrue(
                    np.all(
                        np.logical_or(
                            np.isclose(out - angles, 0),
                            np.isclose(out + angles, 0)
                            )
                        ),
                    msg="Failed for convention {}, axis type {}".format(
                        convention, axis_type))
