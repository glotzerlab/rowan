"""Test the simple unary operator of the quaternion package"""
from __future__ import division, print_function, absolute_import

import unittest
import numpy as np

import hamilton as quaternion

zero = np.array([0, 0, 0, 0])

class TestSimple(unittest.TestCase):
    def test_conjugate(self):
        """Test quaternion conjugation"""
        np.random.seed(0)
        shapes = [(4, ), (5, 4), (5, 5, 4), (5, 5, 5, 4)]
        for shape in shapes:
            quats = np.random.random_sample(shape)
            quats_conj = quats.copy()
            quats_conj[..., 1:] *= -1
            self.assertTrue(np.all(quaternion.conjugate(quats) == quats_conj))

    def test_norm(self):
        """Test quaternion norm"""
        np.random.seed(0)
        shapes = [(4, ), (5, 4), (5, 5, 4), (5, 5, 5, 4)]
        for shape in shapes:
            quats = np.random.random_sample(shape)
            norms = np.linalg.norm(quats, axis=-1)
            self.assertTrue(np.all(quaternion.norm(quats) == norms))

    def test_normalize(self):
        """Test quaternion normalize"""
        np.random.seed(0)
        shapes = [(4, ), (5, 4), (5, 5, 4), (5, 5, 5, 4)]
        for shape in shapes:
            quats = np.random.random_sample(shape)
            norms = np.linalg.norm(quats, axis=-1)
            self.assertTrue(np.all(quaternion.normalize(quats)
                                   == quats / norms[..., np.newaxis]))

    def test_equal(self):
        """Test quaternion equality"""
        np.random.seed(0)
        shapes = [(4, ), (5, 4), (5, 5, 4), (5, 5, 5, 4)]
        for shape in shapes:
            quats = np.random.random_sample(shape)
            self.assertTrue(
                    quaternion.equal(quats, quats).shape ==
                    quats.shape[:-1])
            self.assertTrue(np.all(quaternion.equal(quats, quats)))
            self.assertFalse(
                    np.any(
                        quaternion.equal(quats, 0)
                        )
                    )

    def test_not_equal(self):
        """Test quaternion inequality"""
        np.random.seed(0)
        shapes = [(4, ), (5, 4), (5, 5, 4), (5, 5, 5, 4)]
        for shape in shapes:
            quats = np.random.random_sample(shape)
            self.assertTrue(
                    quaternion.not_equal(quats, quats).shape ==
                    quats.shape[:-1])
            self.assertFalse(np.all(quaternion.not_equal(quats, quats)))
            self.assertTrue(
                    np.any(
                        quaternion.not_equal(quats, 0)
                        )
                    )

    def test_allclose(self):
        """Test all quaternion closeness"""
        np.random.seed(0)
        shapes = [(4, ), (5, 4), (5, 5, 4), (5, 5, 5, 4)]
        for shape in shapes:
            quats = np.random.random_sample(shape)
            self.assertTrue(quaternion.allclose(quats, quats))
            self.assertTrue(quaternion.allclose(quats, quats-1e-8))

    def test_isclose(self):
        """Test element-wise quaternion closeness"""
        np.random.seed(0)
        shapes = [(4, ), (5, 4), (5, 5, 4), (5, 5, 5, 4)]
        for shape in shapes:
            quats = np.random.random_sample(shape)
            self.assertTrue(
                    quaternion.isclose(quats, quats).shape ==
                    quats.shape[:-1])
            self.assertTrue(np.all(quaternion.isclose(quats, quats)))
            self.assertTrue(np.all(quaternion.isclose(quats, quats-1e-8)))

    def test_isfinite(self):
        """Test quaternion finiteness"""
        x = np.array([np.inf]*4)
        self.assertFalse(quaternion.isfinite(x))
        x[1:] = 0
        self.assertFalse(quaternion.isfinite(x))
        self.assertTrue(quaternion.isfinite(zero))

        np.random.seed(0)
        shapes = [(4, ), (5, 4), (5, 5, 4), (5, 5, 5, 4)]
        for shape in shapes:
            quats = np.random.random_sample(shape)
            self.assertTrue(
                    quaternion.isfinite(quats).shape ==
                    quats.shape[:-1])
            self.assertTrue(np.all(quaternion.isfinite(quats)))

    def test_isinf(self):
        """Test quaternion infiniteness"""
        x = np.array([np.inf]*4)
        self.assertTrue(quaternion.isinf(x))
        x[1:] = 0
        self.assertTrue(quaternion.isinf(x))
        self.assertFalse(quaternion.isinf(zero))

        np.random.seed(0)
        shapes = [(4, ), (5, 4), (5, 5, 4), (5, 5, 5, 4)]
        for shape in shapes:
            quats = np.random.random_sample(shape)
            self.assertTrue(
                    quaternion.isinf(quats).shape ==
                    quats.shape[:-1])
            self.assertTrue(np.all(np.logical_not(quaternion.isinf(quats))))

    def test_isnan(self):
        """Test quaternions being of numeric type"""
        x = np.array([np.nan]*4)
        self.assertTrue(quaternion.isnan(x))
        x[1:] = 0
        self.assertTrue(quaternion.isnan(x))
        self.assertFalse(quaternion.isnan(zero))

        np.random.seed(0)
        shapes = [(4, ), (5, 4), (5, 5, 4), (5, 5, 5, 4)]
        for shape in shapes:
            quats = np.random.random_sample(shape)
            self.assertTrue(
                    quaternion.isnan(quats).shape ==
                    quats.shape[:-1])
            self.assertTrue(not np.any(quaternion.isnan(quats)))
