"""Test converting quaternions to and from axis-angle representation"""
from __future__ import division, print_function, absolute_import

import unittest
import numpy as np

import hamilton as quaternion

zero = np.array([0, 0, 0, 0])
one = np.array([1, 0, 0, 0])
half = np.array([0.5, 0.5, 0.5, 0.5])


class TestAxisAngle(unittest.TestCase):
    """Test axis angle conversions"""

    def test_from_axis_angle(self):
        self.assertTrue(
            np.allclose(quaternion.from_axis_angle(
                    np.array([1, 0, 0]),
                    0),
                    np.array((1, 0, 0, 0))
                    )
            )

        self.assertTrue(
            np.allclose(quaternion.from_axis_angle(
                    np.array([1, 0, 0]),
                    np.pi/2),
                    np.array((np.sqrt(2)/2, np.sqrt(2)/2, 0, 0))
                    )
            )

    def test_to_axis_angle(self):
        axes, angles = quaternion.to_axis_angle(
                np.array((np.sqrt(2)/2, np.sqrt(2)/2, 0, 0)))
        self.assertTrue(np.allclose(axes, np.array([1, 0, 0])))
        self.assertTrue(np.allclose(angles, np.pi/2))
