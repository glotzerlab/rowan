"""Test function to rotate vector onto vector"""
from __future__ import division, print_function, absolute_import

import unittest
import numpy as np

import hamilton as quaternion


class TestVectorVector(unittest.TestCase):
    def test_vector_vector_rotation(self):
        """Test finding quaternion to rotate a vector onto another vector"""
        vec1 = np.array([0, 0, 1])
        vec2 = np.array([1, 0, 0])
        quat = quaternion.vector_vector_rotation(vec1, vec2)
        self.assertTrue(np.allclose(quat, np.array(
            [[0, np.sqrt(2)/2, 0, np.sqrt(2)/2]])))

        vec1 = np.array([[0, 0, 1]])
        vec2 = np.array([[1, 0, 0]])
        quat = quaternion.vector_vector_rotation(vec1, vec2)
        self.assertTrue(np.allclose(quat, np.array(
            [[0, np.sqrt(2)/2, 0, np.sqrt(2)/2]])))
