"""Test functions to compute quaternion distance."""

import os
import unittest

import numpy as np

from rowan import geometry, normalize

zero = np.array([0, 0, 0, 0])
one = np.array([1, 0, 0, 0])
inf_array = np.array([-np.inf, 0, 0, 0])


class TestDistance(unittest.TestCase):
    """Test distance metrics."""

    def setUp(self):
        """Get the data file used for all methods."""
        self.saved_data = np.load(
            os.path.join(os.path.dirname(__file__), "files/test_geometry.npz"),
        )

    def test_distance(self):
        """Test the simple distance metric."""
        assert geometry.distance(one, one) == 0
        assert geometry.distance(zero, zero) == 0
        assert geometry.distance(zero, one) == 1

        x1 = np.stack((zero, one))
        x2 = np.stack((one, zero))

        assert np.all(geometry.distance(x1, x1) == 0)
        assert np.all(geometry.distance(x1, x2) == 1)

        my_ans = geometry.distance(self.saved_data["p"], self.saved_data["q"])
        assert np.allclose(my_ans, self.saved_data["distance"])

    def test_sym_distance(self):
        """Test the symmetric distance metric."""
        assert geometry.sym_distance(one, one) == 0
        assert geometry.sym_distance(zero, zero) == 0
        assert geometry.sym_distance(zero, one) == 1

        x1 = np.stack((zero, one))
        x2 = np.stack((one, zero))

        assert np.all(geometry.sym_distance(x1, x1) == 0)
        assert np.all(geometry.sym_distance(x1, x2) == 1)

        my_ans = geometry.sym_distance(self.saved_data["p"], self.saved_data["q"])
        assert np.allclose(self.saved_data["sym_distance"], my_ans)

    def test_riemann_exp_map(self):
        """Test computation of the Riemannian exponential map."""
        assert np.all(geometry.riemann_exp_map(zero, zero) == 0)
        assert np.all(geometry.riemann_exp_map(one, zero) == one)
        assert np.allclose(geometry.riemann_exp_map(one, one), [np.exp(1), 0, 0, 0])

        x1 = np.stack((zero, one))
        x2 = np.stack((one, zero))

        assert np.all(geometry.riemann_exp_map(x1, x2) == x1)
        assert np.all(geometry.riemann_exp_map(x2, x1) == x2)

        my_ans = geometry.riemann_exp_map(self.saved_data["p"], self.saved_data["q"])
        assert np.allclose(self.saved_data["exp_map"], my_ans)

    def test_riemann_log_map(self):
        """Test computation of the Riemannian logarithmic map."""
        assert np.all(geometry.riemann_log_map(zero, zero) == inf_array)
        assert np.all(geometry.riemann_log_map(one, zero) == inf_array)
        assert np.allclose(geometry.riemann_log_map(one, one), [np.log(1), 0, 0, 0])

        x1 = np.stack((zero, one))
        x2 = np.stack((one, zero))

        assert np.all(geometry.riemann_log_map(x1, x2) == inf_array)
        assert np.all(geometry.riemann_log_map(x2, x1) == inf_array)

        my_ans = geometry.riemann_log_map(self.saved_data["p"], self.saved_data["q"])
        assert np.allclose(self.saved_data["log_map"], my_ans)

    def test_intrinsic_distance(self):
        """Test computation of the intrinsic distance."""
        assert geometry.intrinsic_distance(zero, zero) == np.inf
        assert geometry.intrinsic_distance(one, zero) == np.inf
        assert np.allclose(geometry.intrinsic_distance(one, one), 0)

        x1 = np.stack((zero, one))
        x2 = np.stack((one, zero))

        assert np.all(geometry.intrinsic_distance(x1, x2) == np.inf)
        assert np.all(geometry.intrinsic_distance(x2, x1) == np.inf)

        my_ans = geometry.intrinsic_distance(self.saved_data["p"], self.saved_data["q"])
        assert np.allclose(self.saved_data["intrinsic_distance"], my_ans)

    def test_sym_intrinsic_distance(self):
        """Test computation of the symmetric intrinsic distance."""
        assert geometry.sym_intrinsic_distance(zero, zero) == np.inf
        assert geometry.sym_intrinsic_distance(one, zero) == np.inf
        assert np.allclose(geometry.sym_intrinsic_distance(one, one), 0)

        x1 = np.stack((zero, one))
        x2 = np.stack((one, zero))

        assert np.all(geometry.sym_intrinsic_distance(x1, x2) == np.inf)
        assert np.all(geometry.sym_intrinsic_distance(x2, x1) == np.inf)

        my_ans = geometry.sym_intrinsic_distance(
            self.saved_data["p"],
            self.saved_data["q"],
        )
        assert np.allclose(self.saved_data["sym_intrinsic_distance"], my_ans)

    def test_angle(self):
        """Test computation of angles."""
        assert geometry.angle(one) == 0

        p = normalize(self.saved_data["p"])
        my_ans = geometry.angle(p)
        assert np.allclose(self.saved_data["angle"], my_ans)
