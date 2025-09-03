"""Test converting quaternions to and from rotation matrices."""

import os
import unittest

import numpy as np
import pytest

import rowan

zero = np.array([0, 0, 0, 0])
one = np.array([1, 0, 0, 0])
half = np.array([0.5, 0.5, 0.5, 0.5])

# Load test files
TESTDATA_FILENAME = os.path.join(os.path.dirname(__file__), "files/test_arrays.npz")
with np.load(TESTDATA_FILENAME) as data:
    input1 = data["input1"]
    vector_inputs = data["vector_inputs"]


class TestMatrix(unittest.TestCase):
    """Test rotation matrix conversions."""

    def test_from_matrix(self):
        """Test conversion from a matrix to a quaternion."""
        assert np.all(rowan.from_matrix(np.eye(3)) == one)

        with pytest.raises(ValueError):
            assert np.allclose(rowan.from_matrix(2 * np.eye(3)))

        mat = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

        assert np.logical_or(
            np.allclose(rowan.from_matrix(mat), half),
            np.allclose(rowan.from_matrix(mat), -half),
        )

        mat = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
        v = np.copy(half)
        v[3] *= -1
        assert np.allclose(rowan.from_matrix(mat), v)

    def test_to_matrix(self):
        """Test conversion from a quaternion to a matrix."""
        v = np.copy(zero)
        with pytest.raises(ZeroDivisionError):
            rowan.to_matrix(v)

        v = 2 * np.ones(4)
        with pytest.raises(ValueError):
            rowan.to_matrix(v)

        v = np.copy(one)
        assert np.all(rowan.to_matrix(v) == np.eye(3))

        v = np.copy(half)
        assert np.allclose(
            rowan.to_matrix(v), np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        )

        v[3] *= -1
        assert np.allclose(
            rowan.to_matrix(v), np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
        )

    def test_to_from_matrix(self):
        """Test conversion from a quaternion to a matrix and back."""
        # The equality is only guaranteed up to a sign
        converted = rowan.from_matrix(rowan.to_matrix(input1))
        assert np.all(
            np.logical_or(
                np.isclose(input1 - converted, 0), np.isclose(input1 + converted, 0)
            )
        )

    def test_rotation(self):
        """Test that a quaternion and its converted matrix perform the same rotation."""
        quat_rotated = rowan.rotate(input1, vector_inputs)

        matrices = rowan.to_matrix(input1)
        matrix_rotated = np.einsum("ijk,ki->ij", matrices, vector_inputs.T)
        assert np.allclose(matrix_rotated, quat_rotated)
