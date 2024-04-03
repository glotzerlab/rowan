"""Test the simple unary operator of the quaternion package."""

import unittest

import numpy as np

import rowan

zero = np.array([0, 0, 0, 0])


class TestSimple(unittest.TestCase):
    """Test simple quaternion operations in the core of the package."""

    def test_conjugate(self):
        """Test quaternion conjugation."""
        np.random.seed(0)
        shapes = [(4,), (5, 4), (5, 5, 4), (5, 5, 5, 4)]
        for shape in shapes:
            quats = np.random.random_sample(shape)
            quats_conj = quats.copy()
            quats_conj[..., 1:] *= -1
            self.assertTrue(np.all(rowan.conjugate(quats) == quats_conj))

    def test_inverse(self):
        """Test quaternion inverse."""
        np.random.seed(0)
        shapes = [(4,), (5, 4), (5, 5, 4), (5, 5, 5, 4)]
        for shape in shapes:
            quats = np.random.random_sample(shape)
            quats_conj = quats.copy()
            quats_conj[..., 1:] *= -1
            quats_conj /= rowan.norm(quats)[..., np.newaxis] ** 2
            self.assertTrue(np.allclose(rowan.inverse(quats), quats_conj))

    def test_norm(self):
        """Test quaternion norm."""
        np.random.seed(0)
        shapes = [(4,), (5, 4), (5, 5, 4), (5, 5, 5, 4)]
        for shape in shapes:
            quats = np.random.random_sample(shape)
            norms = np.linalg.norm(quats, axis=-1)
            self.assertTrue(np.all(rowan.norm(quats) == norms))

    def test_normalize(self):
        """Test quaternion normalize."""
        np.random.seed(0)
        shapes = [(4,), (5, 4), (5, 5, 4), (5, 5, 5, 4)]
        for shape in shapes:
            quats = np.random.random_sample(shape)
            norms = np.linalg.norm(quats, axis=-1)
            self.assertTrue(
                np.all(rowan.normalize(quats) == quats / norms[..., np.newaxis]),
            )

    def test_equal(self):
        """Test quaternion equality."""
        np.random.seed(0)
        shapes = [(4,), (5, 4), (5, 5, 4), (5, 5, 5, 4)]
        for shape in shapes:
            quats = np.random.random_sample(shape)
            self.assertTrue(rowan.equal(quats, quats).shape == quats.shape[:-1])
            self.assertTrue(np.all(rowan.equal(quats, quats)))
            self.assertFalse(np.any(rowan.equal(quats, 0)))

    def test_not_equal(self):
        """Test quaternion inequality."""
        np.random.seed(0)
        shapes = [(4,), (5, 4), (5, 5, 4), (5, 5, 5, 4)]
        for shape in shapes:
            quats = np.random.random_sample(shape)
            self.assertTrue(rowan.not_equal(quats, quats).shape == quats.shape[:-1])
            self.assertFalse(np.all(rowan.not_equal(quats, quats)))
            self.assertTrue(np.any(rowan.not_equal(quats, 0)))

    def test_allclose(self):
        """Test all quaternion closeness."""
        np.random.seed(0)
        shapes = [(4,), (5, 4), (5, 5, 4), (5, 5, 5, 4)]
        for shape in shapes:
            quats = np.random.random_sample(shape)
            self.assertTrue(rowan.allclose(quats, quats))
            self.assertTrue(rowan.allclose(quats, quats - 1e-8))

    def test_isclose(self):
        """Test element-wise quaternion closeness."""
        np.random.seed(0)
        shapes = [(4,), (5, 4), (5, 5, 4), (5, 5, 5, 4)]
        for shape in shapes:
            quats = np.random.random_sample(shape)
            self.assertTrue(rowan.isclose(quats, quats).shape == quats.shape[:-1])
            self.assertTrue(np.all(rowan.isclose(quats, quats)))
            self.assertTrue(np.all(rowan.isclose(quats, quats - 1e-8)))

    def test_isfinite(self):
        """Test quaternion finiteness."""
        x = np.array([np.inf] * 4)
        self.assertFalse(rowan.isfinite(x))
        x[1:] = 0
        self.assertFalse(rowan.isfinite(x))
        self.assertTrue(rowan.isfinite(zero))

        np.random.seed(0)
        shapes = [(4,), (5, 4), (5, 5, 4), (5, 5, 5, 4)]
        for shape in shapes:
            quats = np.random.random_sample(shape)
            self.assertTrue(rowan.isfinite(quats).shape == quats.shape[:-1])
            self.assertTrue(np.all(rowan.isfinite(quats)))

    def test_isinf(self):
        """Test quaternion infiniteness."""
        x = np.array([np.inf] * 4)
        self.assertTrue(rowan.isinf(x))
        x[1:] = 0
        self.assertTrue(rowan.isinf(x))
        self.assertFalse(rowan.isinf(zero))

        np.random.seed(0)
        shapes = [(4,), (5, 4), (5, 5, 4), (5, 5, 5, 4)]
        for shape in shapes:
            quats = np.random.random_sample(shape)
            self.assertTrue(rowan.isinf(quats).shape == quats.shape[:-1])
            self.assertTrue(np.all(np.logical_not(rowan.isinf(quats))))

    def test_isnan(self):
        """Test quaternions being of numeric type."""
        x = np.array([np.nan] * 4)
        self.assertTrue(rowan.isnan(x))
        x[1:] = 0
        self.assertTrue(rowan.isnan(x))
        self.assertFalse(rowan.isnan(zero))

        np.random.seed(0)
        shapes = [(4,), (5, 4), (5, 5, 4), (5, 5, 5, 4)]
        for shape in shapes:
            quats = np.random.random_sample(shape)
            self.assertTrue(rowan.isnan(quats).shape == quats.shape[:-1])
            self.assertTrue(not np.any(rowan.isnan(quats)))
