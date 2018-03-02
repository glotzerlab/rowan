"""Test the basic functions of the quaternion package that can be tested easily"""
from __future__ import division, print_function, absolute_import

import unittest
import numpy as np

import hamilton as quaternion


class TestSimple(unittest.TestCase):
    def test_conjugate(self):
        """Test quaternion conjugation"""
        shapes = [(4, ), (5, 4), (5, 5, 4), (5, 5, 5, 4)]
        for shape in shapes:
            quats = np.random.random_sample(shape)
            quats_conj = quats.copy()
            quats_conj[..., 1:] *= -1
            self.assertTrue(np.all(quaternion.conjugate(quats) == quats_conj))

    def test_norm(self):
        """Test quaternion norm"""
        shapes = [(4, ), (5, 4), (5, 5, 4), (5, 5, 5, 4)]
        for shape in shapes:
            quats = np.random.random_sample(shape)
            norms = np.linalg.norm(quats, axis=-1)
            self.assertTrue(np.all(quaternion.norm(quats) == norms))

    def test_normalize(self):
        """Test quaternion normalize"""
        shapes = [(4, ), (5, 4), (5, 5, 4), (5, 5, 5, 4)]
        for shape in shapes:
            quats = np.random.random_sample(shape)
            norms = np.linalg.norm(quats, axis=-1)
            self.assertTrue(np.all(quaternion.normalize(quats)
                                   == quats / norms[..., np.newaxis]))
