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

    def test_mean(self, N=128):
        """Test mean taken between quaternions."""
        rng = np.random.default_rng(seed=0)
        qs = rowan.random.rand(N)
        # Verify mean of one quaternion (or duplicates of the same quat) == q_input
        for q in qs:
            for n in [1, 2, 3]:
                assert rowan.isclose(q, rowan.mean([q] * n)) or rowan.isclose(
                    q, -rowan.mean([q] * n)
                )
                assert rowan.isclose(
                    q, rowan.mean([q] * n, weights=np.ones(n))
                ) or rowan.isclose(q, -rowan.mean([q] * n, weights=np.ones(n)))

        def mean_two_quats(q0, q1, w0=1, w1=1):
            """Compute the maximum-likelihood mean of two quaternions in closed form."""
            z = np.sqrt(np.square(w0 - w1) + 4 * w0 * w1 * np.square(np.dot(q0, q1)))
            s0 = np.sqrt((w0 * (w0 - w1 + z)) / (z * (w0 + w1 + z)))
            s1 = np.sqrt((w1 * (w1 - w0 + z)) / (z * (w0 + w1 + z)))
            return s0 * q0 + np.sign(np.dot(q0, q1)) * s1 * q1

        # Split list of quaternions in half and zip into pairs
        for w in [np.ones(2), rng.random(2)]:
            for q0, q1 in zip(qs[: N // 2, :], qs[N // 2 :, :]):
                assert rowan.isclose(
                    rowan.mean([q0, q1], weights=w), mean_two_quats(q0, q1, w[0], w[1])
                ) or rowan.isclose(
                    rowan.mean([q0, q1], weights=w), -mean_two_quats(q0, q1, w[0], w[1])
                )
