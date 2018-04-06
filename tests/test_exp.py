"""Test exponential, log, and powers of quaternions"""
from __future__ import division, print_function, absolute_import

import numpy as np
import os

import hamilton as quaternion
import unittest


class TestExp(unittest.TestCase):
    """Test exponential function"""

    def test_exp(self):
        """Ensure that quaternion exponential behaves correctly"""
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
