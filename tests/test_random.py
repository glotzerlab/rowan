"""Test the generation of random quaternions."""

import unittest

import numpy as np

from rowan import norm
from rowan import random as random


class TestRandom(unittest.TestCase):
    """Test the generation of random quaternions."""

    def test_random(self):
        """Generation from various args."""
        s = (3, 4)
        np.random.seed(0)
        q = random.rand(s[0], s[1])
        assert q.shape == s + (4,)
        assert np.allclose(norm(q), 1)

        q = random.rand()
        assert q.shape == (4,)

    def test_random_sample(self):
        """Generation with tuple."""
        s = (3, 4)
        np.random.seed(0)
        q = random.random_sample(s)
        assert q.shape == s + (4,)
        assert np.allclose(norm(q), 1)

        q = random.random_sample()
        assert q.shape == (4,)
