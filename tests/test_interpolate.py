"""Test functions to interpolate between quaternion."""
from __future__ import absolute_import, division, print_function

import unittest

import numpy as np

import rowan
from rowan import interpolate

zero = np.array([0, 0, 0, 0])
one = np.array([1, 0, 0, 0])
root_two = np.array([np.sqrt(2) / 2, np.sqrt(2) / 2, 0, 0])


class TestInterpolate(unittest.TestCase):
    """Test quaternion interpolation."""

    def test_slerp(self):
        """Test spherical linear interpolation."""
        self.assertTrue(np.all(interpolate.slerp(one, one, 0) == one))
        self.assertTrue(np.all(interpolate.slerp(one, one, 1) == one))
        self.assertTrue(np.all(interpolate.slerp(one, one, 0.5) == one))

        ans = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.70710678, 0.70710678, 0.0, 0.0],
                [0.92387953, 0.38268343, 0.0, 0.0],
            ]
        )
        self.assertTrue(np.allclose(interpolate.slerp(one, root_two, 0), ans[0, :]))
        self.assertTrue(np.allclose(interpolate.slerp(one, root_two, 1), ans[1, :]))
        self.assertTrue(np.allclose(interpolate.slerp(one, root_two, 0.5), ans[2, :]))
        self.assertTrue(np.allclose(interpolate.slerp(one, root_two, [0, 1, 0.5]), ans))

        tmp = np.stack((one, root_two), axis=0)
        test1 = np.stack((tmp, tmp), axis=0)
        self.assertTrue(np.all(interpolate.slerp(test1, test1, 0) == test1))
        self.assertTrue(np.allclose(interpolate.slerp(test1, test1, 1), test1))
        self.assertTrue(np.all(interpolate.slerp(test1, test1, 0.5) == test1))
        self.assertTrue(
            np.allclose(
                interpolate.slerp(
                    test1, test1, np.array([0, 1, 0.5])[:, np.newaxis, np.newaxis]
                ),
                np.stack((test1, test1, test1)),
            )
        )

    def test_slerp_prime(self):
        """Test spherical linear interpolation derivative."""
        self.assertTrue(np.all(interpolate.slerp_prime(one, one, 0) == zero))
        self.assertTrue(np.all(interpolate.slerp_prime(one, one, 1) == zero))
        self.assertTrue(np.all(interpolate.slerp_prime(one, one, 0.5) == zero))

        self.assertTrue(
            np.allclose(
                interpolate.slerp_prime(root_two, one, 0.5),
                rowan.multiply(
                    interpolate.slerp(root_two, one, 0.5),
                    rowan.log(rowan.multiply(rowan.conjugate(root_two), one)),
                ),
            )
        )

    def test_squad(self):
        """Test spherical quadratic interpolation."""
        self.assertTrue(np.all(interpolate.squad(one, one, one, one, 0) == one))
        self.assertTrue(
            np.allclose(interpolate.squad(one, one, one, root_two, 1), root_two)
        )
        self.assertTrue(
            np.allclose(
                interpolate.squad(one, one, one, root_two, 0.5),
                np.array([0.98078528, 0.19509032, 0, 0.0]),
            )
        )
