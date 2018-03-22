"""Test the generation of random quaternions"""
from __future__ import division, print_function, absolute_import

import unittest
import numpy as np

from hamilton import random as random


class TestRandom(unittest.TestCase):
    """Test the generation of random quaternions"""

    def test_random(self):
        """Generation from various args"""
        print(random.rand(3, 4))

    def test_random_sample(self):
        """Generation with tuple"""
        pass
