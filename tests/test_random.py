"""Test the generation of random quaternions"""
from __future__ import division, print_function, absolute_import

import unittest
import numpy as np

from hamilton import random as random
from hamilton import norm


class TestRandom(unittest.TestCase):
    """Test the generation of random quaternions"""

    def test_random(self):
        """Generation from various args"""
        s = (3, 4)
        np.random.seed(0)
        q = random.rand(s[0], s[1])
        self.assertTrue(q.shape == s + (4,))
        self.assertTrue(np.allclose(norm(q), 1))

    def test_random_sample(self):
        """Generation with tuple"""
        s = (3, 4)
        np.random.seed(0)
        q = random.random_sample(s)
        self.assertTrue(q.shape == s + (4,))
        self.assertTrue(np.allclose(norm(q), 1))
