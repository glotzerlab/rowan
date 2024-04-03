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
            assert np.all(rowan.conjugate(quats) == quats_conj)

    def test_inverse(self):
        """Test quaternion inverse."""
        np.random.seed(0)
        shapes = [(4,), (5, 4), (5, 5, 4), (5, 5, 5, 4)]
        for shape in shapes:
            quats = np.random.random_sample(shape)
            quats_conj = quats.copy()
            quats_conj[..., 1:] *= -1
            quats_conj /= rowan.norm(quats)[..., np.newaxis] ** 2
            assert np.allclose(rowan.inverse(quats), quats_conj)

    def test_norm(self):
        """Test quaternion norm."""
        np.random.seed(0)
        shapes = [(4,), (5, 4), (5, 5, 4), (5, 5, 5, 4)]
        for shape in shapes:
            quats = np.random.random_sample(shape)
            norms = np.linalg.norm(quats, axis=-1)
            assert np.all(rowan.norm(quats) == norms)

    def test_normalize(self):
        """Test quaternion normalize."""
        np.random.seed(0)
        shapes = [(4,), (5, 4), (5, 5, 4), (5, 5, 5, 4)]
        for shape in shapes:
            quats = np.random.random_sample(shape)
            norms = np.linalg.norm(quats, axis=-1)
            assert np.all(rowan.normalize(quats) == quats / norms[..., np.newaxis])

    def test_equal(self):
        """Test quaternion equality."""
        np.random.seed(0)
        shapes = [(4,), (5, 4), (5, 5, 4), (5, 5, 5, 4)]
        for shape in shapes:
            quats = np.random.random_sample(shape)
            assert rowan.equal(quats, quats).shape == quats.shape[:-1]
            assert np.all(rowan.equal(quats, quats))
            assert not np.any(rowan.equal(quats, 0))

    def test_not_equal(self):
        """Test quaternion inequality."""
        np.random.seed(0)
        shapes = [(4,), (5, 4), (5, 5, 4), (5, 5, 5, 4)]
        for shape in shapes:
            quats = np.random.random_sample(shape)
            assert rowan.not_equal(quats, quats).shape == quats.shape[:-1]
            assert not np.all(rowan.not_equal(quats, quats))
            assert np.any(rowan.not_equal(quats, 0))

    def test_allclose(self):
        """Test all quaternion closeness."""
        np.random.seed(0)
        shapes = [(4,), (5, 4), (5, 5, 4), (5, 5, 5, 4)]
        for shape in shapes:
            quats = np.random.random_sample(shape)
            assert rowan.allclose(quats, quats)
            assert rowan.allclose(quats, quats - 1e-08)

    def test_isclose(self):
        """Test element-wise quaternion closeness."""
        np.random.seed(0)
        shapes = [(4,), (5, 4), (5, 5, 4), (5, 5, 5, 4)]
        for shape in shapes:
            quats = np.random.random_sample(shape)
            assert rowan.isclose(quats, quats).shape == quats.shape[:-1]
            assert np.all(rowan.isclose(quats, quats))
            assert np.all(rowan.isclose(quats, quats - 1e-08))

    def test_isfinite(self):
        """Test quaternion finiteness."""
        x = np.array([np.inf] * 4)
        assert not rowan.isfinite(x)
        x[1:] = 0
        assert not rowan.isfinite(x)
        assert rowan.isfinite(zero)

        np.random.seed(0)
        shapes = [(4,), (5, 4), (5, 5, 4), (5, 5, 5, 4)]
        for shape in shapes:
            quats = np.random.random_sample(shape)
            assert rowan.isfinite(quats).shape == quats.shape[:-1]
            assert np.all(rowan.isfinite(quats))

    def test_isinf(self):
        """Test quaternion infiniteness."""
        x = np.array([np.inf] * 4)
        assert rowan.isinf(x)
        x[1:] = 0
        assert rowan.isinf(x)
        assert not rowan.isinf(zero)

        np.random.seed(0)
        shapes = [(4,), (5, 4), (5, 5, 4), (5, 5, 5, 4)]
        for shape in shapes:
            quats = np.random.random_sample(shape)
            assert rowan.isinf(quats).shape == quats.shape[:-1]
            assert np.all(np.logical_not(rowan.isinf(quats)))

    def test_isnan(self):
        """Test quaternions being of numeric type."""
        x = np.array([np.nan] * 4)
        assert rowan.isnan(x)
        x[1:] = 0
        assert rowan.isnan(x)
        assert not rowan.isnan(zero)

        np.random.seed(0)
        shapes = [(4,), (5, 4), (5, 5, 4), (5, 5, 5, 4)]
        for shape in shapes:
            quats = np.random.random_sample(shape)
            assert rowan.isnan(quats).shape == quats.shape[:-1]
            assert not np.any(rowan.isnan(quats))
