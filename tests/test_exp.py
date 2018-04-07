"""Test exponential, log, and powers of quaternions"""
from __future__ import division, print_function, absolute_import

import numpy as np
import os

import hamilton as quaternion
import unittest

one = np.array([1, 0, 0, 0])
zero = np.array([0, 0, 0, 0])


class TestExp(unittest.TestCase):
    """Test exponential function"""

    def test_exp(self):
        """Ensure that quaternion exponential behaves correctly"""
        self.assertTrue(
                np.all(quaternion.exp(one) ==
                       np.array([np.exp(1), 0, 0, 0])
                       )
                )
        x = np.array([0, 1, 1, 1])
        self.assertTrue(
                np.allclose(
                    quaternion.exp(x),
                    np.array([-0.16055654, 0.5698601, 0.5698601, 0.5698601])
                    )
                )
        self.assertTrue(np.all(quaternion.exp(zero) == one))

        np.random.seed(0)
        shapes = [(4,), (1, 4), (3, 4, 4), (12, 7, 3, 4)]
        answers = np.load(os.path.join(
            os.path.dirname(__file__),
            'files/test_exp.npz'))
        for shape in shapes:
            x = np.random.random_sample(shape)
            self.assertTrue(
                    np.allclose(
                        quaternion.exp(x), answers[str(shape)]
                        ),
                    msg="Failed for shape {}".format(shape))

    def test_log(self):
        """Ensure that quaternion logarithm behaves correctly"""
        x = np.array([1, 0, 0, 0])
        self.assertTrue(np.all(quaternion.log(x) == zero))
        x = np.array([0, 0, 0, 0])
        self.assertTrue(
                np.all(
                    quaternion.log(x) ==
                    np.array([-np.inf, 0, 0, 0])
                    )
                )

        np.random.seed(0)
        shapes = [(4,), (1, 4), (3, 4, 4), (12, 7, 3, 4)]
        answers = np.load(os.path.join(
            os.path.dirname(__file__),
            'files/test_log.npz'))
        for shape in shapes:
            x = np.random.random_sample(shape)
            self.assertTrue(
                    np.allclose(
                        quaternion.log(x), answers[str(shape)]
                        ),
                    msg="Failed for shape {}".format(shape))
