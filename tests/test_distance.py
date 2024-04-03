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
            os.path.join(os.path.dirname(__file__), "files/test_geometry.npz")
        )

    def test_distance(self):
        """Test the simple distance metric."""
        self.assertTrue(geometry.distance(one, one) == 0)
        self.assertTrue(geometry.distance(zero, zero) == 0)
        self.assertTrue(geometry.distance(zero, one) == 1)

        x1 = np.stack((zero, one))
        x2 = np.stack((one, zero))

        self.assertTrue(np.all(geometry.distance(x1, x1) == 0))
        self.assertTrue(np.all(geometry.distance(x1, x2) == 1))

        my_ans = geometry.distance(self.saved_data["p"], self.saved_data["q"])
        self.assertTrue(np.allclose(my_ans, self.saved_data["distance"]))

    def test_sym_distance(self):
        """Test the symmetric distance metric."""
        self.assertTrue(geometry.sym_distance(one, one) == 0)
        self.assertTrue(geometry.sym_distance(zero, zero) == 0)
        self.assertTrue(geometry.sym_distance(zero, one) == 1)

        x1 = np.stack((zero, one))
        x2 = np.stack((one, zero))

        self.assertTrue(np.all(geometry.sym_distance(x1, x1) == 0))
        self.assertTrue(np.all(geometry.sym_distance(x1, x2) == 1))

        my_ans = geometry.sym_distance(self.saved_data["p"], self.saved_data["q"])
        self.assertTrue(np.allclose(self.saved_data["sym_distance"], my_ans))

    def test_riemann_exp_map(self):
        """Test computation of the Riemannian exponential map."""
        self.assertTrue(np.all(geometry.riemann_exp_map(zero, zero) == 0))
        self.assertTrue(np.all(geometry.riemann_exp_map(one, zero) == one))
        self.assertTrue(
            np.allclose(geometry.riemann_exp_map(one, one), [np.exp(1), 0, 0, 0])
        )

        x1 = np.stack((zero, one))
        x2 = np.stack((one, zero))

        self.assertTrue(np.all(geometry.riemann_exp_map(x1, x2) == x1))
        self.assertTrue(np.all(geometry.riemann_exp_map(x2, x1) == x2))

        my_ans = geometry.riemann_exp_map(self.saved_data["p"], self.saved_data["q"])
        self.assertTrue(np.allclose(self.saved_data["exp_map"], my_ans))

    def test_riemann_log_map(self):
        """Test computation of the Riemannian logarithmic map."""
        self.assertTrue(np.all(geometry.riemann_log_map(zero, zero) == inf_array))
        self.assertTrue(np.all(geometry.riemann_log_map(one, zero) == inf_array))
        self.assertTrue(
            np.allclose(geometry.riemann_log_map(one, one), [np.log(1), 0, 0, 0])
        )

        x1 = np.stack((zero, one))
        x2 = np.stack((one, zero))

        self.assertTrue(np.all(geometry.riemann_log_map(x1, x2) == inf_array))
        self.assertTrue(np.all(geometry.riemann_log_map(x2, x1) == inf_array))

        my_ans = geometry.riemann_log_map(self.saved_data["p"], self.saved_data["q"])
        self.assertTrue(np.allclose(self.saved_data["log_map"], my_ans))

    def test_intrinsic_distance(self):
        """Test computation of the intrinsic distance."""
        self.assertTrue(geometry.intrinsic_distance(zero, zero) == np.inf)
        self.assertTrue(geometry.intrinsic_distance(one, zero) == np.inf)
        self.assertTrue(np.allclose(geometry.intrinsic_distance(one, one), 0))

        x1 = np.stack((zero, one))
        x2 = np.stack((one, zero))

        self.assertTrue(np.all(geometry.intrinsic_distance(x1, x2) == np.inf))
        self.assertTrue(np.all(geometry.intrinsic_distance(x2, x1) == np.inf))

        my_ans = geometry.intrinsic_distance(self.saved_data["p"], self.saved_data["q"])
        self.assertTrue(np.allclose(self.saved_data["intrinsic_distance"], my_ans))

    def test_sym_intrinsic_distance(self):
        """Test computation of the symmetric intrinsic distance."""
        self.assertTrue(geometry.sym_intrinsic_distance(zero, zero) == np.inf)
        self.assertTrue(geometry.sym_intrinsic_distance(one, zero) == np.inf)
        self.assertTrue(np.allclose(geometry.sym_intrinsic_distance(one, one), 0))

        x1 = np.stack((zero, one))
        x2 = np.stack((one, zero))

        self.assertTrue(np.all(geometry.sym_intrinsic_distance(x1, x2) == np.inf))
        self.assertTrue(np.all(geometry.sym_intrinsic_distance(x2, x1) == np.inf))

        my_ans = geometry.sym_intrinsic_distance(
            self.saved_data["p"], self.saved_data["q"]
        )
        self.assertTrue(np.allclose(self.saved_data["sym_intrinsic_distance"], my_ans))

    def test_angle(self):
        """Test computation of angles."""
        self.assertTrue(geometry.angle(one) == 0)

        p = normalize(self.saved_data["p"])
        my_ans = geometry.angle(p)
        self.assertTrue(np.allclose(self.saved_data["angle"], my_ans))
