"""Test derivative and integral."""

import unittest

import numpy as np

from rowan import calculus

one = np.array([1, 0, 0, 0])
zero_vec = np.array([0, 0, 0])
one_vec = np.array([1, 0, 0])


class TestCalculus(unittest.TestCase):
    """Test derivatives and integrals."""

    def test_derivative(self):
        """Test differentiation."""
        assert np.all(calculus.derivative(one, zero_vec) == 0)
        assert np.all(calculus.derivative(one, one_vec) == [0, 0.5, 0, 0])

        x = np.array([np.sqrt(2) / 2, np.sqrt(2) / 2, 0, 0])
        v = np.array([0.1, 0.1, 0.1])
        ans = np.array(
            [-0.035355339059327383, 0.035355339059327383, 0.0, 0.070710678118654766],
        )

        assert np.allclose(calculus.derivative(x, v), ans)

    def test_integrate(self):
        """Test integration."""
        assert np.all(calculus.integrate(one, zero_vec, 0) == one)
        assert np.all(calculus.integrate(one, one_vec, 0) == one)
        assert np.all(calculus.integrate(one, zero_vec, 1) == one)
        ans = np.array([0.87758256, 0.47942554, 0.0, 0.0])
        assert np.allclose(calculus.integrate(one, one_vec, 1), ans)

        x = np.array([np.sqrt(2) / 2, np.sqrt(2) / 2, 0, 0])
        v = np.array([0.1, 0.1, 0.1])
        ans = np.array([0.66914563, 0.73976795, 0.07062232, 0])

        assert np.allclose(calculus.integrate(x, v, 1), ans)
