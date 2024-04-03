"""Test the rotation of quaternions for various array sizes."""

import os
import unittest

import numpy as np
import pytest

import rowan

one = np.array([1, 0, 0, 0])

zero_vector = np.array([0, 0, 0])
one_vector = np.array([1, 0, 0])

# Load test files
TESTDATA_FILENAME = os.path.join(os.path.dirname(__file__), "files/test_arrays.npz")
with np.load(TESTDATA_FILENAME) as data:
    input1 = data["input1"]
    input2 = data["input2"]
    stored_rotation = data["rotated_vectors"]
    vector_inputs = data["vector_inputs"]


class TestRotate(unittest.TestCase):
    """Test the core rotation operation."""

    def test_single_quaternion(self):
        """Testing trivial rotations."""
        assert np.all(rowan.rotate(one, one_vector) == one_vector)

    def test_2d_array(self):
        """Rotating sets of vectors by sets of quaternions."""
        ones = np.repeat(one[np.newaxis, :], 10, axis=0)
        one_vectors = np.repeat(one_vector[np.newaxis, :], 10, axis=0)

        # Simple tests
        assert np.all(rowan.rotate(ones, one_vectors) == one_vectors)

        # Complex random array
        assert np.allclose(rowan.rotate(input1, vector_inputs), stored_rotation)

    def test_3d_array(self):
        """Rotating higher dimensional arrays of vectors by arrays of quaternions."""
        num_reps = 20
        expanded_shape = (num_reps // 5, 5, 4)
        expanded_shape_vec = (num_reps // 5, 5, 3)
        ones = np.reshape(
            np.repeat(one[np.newaxis, :], num_reps, axis=0),
            expanded_shape,
        )
        one_vectors = np.reshape(
            np.repeat(one_vector[np.newaxis, :], num_reps, axis=0),
            expanded_shape_vec,
        )

        # Simple tests
        assert np.all(rowan.rotate(ones, one_vectors) == one_vectors)

        # Complex random array
        num_reps = input1.shape[0]
        expanded_shape = (num_reps // 5, 5, 4)
        expanded_shape_vec = (num_reps // 5, 5, 3)
        rotation_result = rowan.rotate(
            np.reshape(input1, expanded_shape),
            np.reshape(vector_inputs, expanded_shape_vec),
        )
        assert np.allclose(
            rotation_result, np.reshape(stored_rotation, expanded_shape_vec)
        )

    def test_broadcast(self):
        """Ensure broadcasting works."""
        # Rotate zero by one, simple shape check
        shape = (45, 3, 13, 4)
        shape_out = (45, 3, 13, 3)
        many_ones = np.zeros(shape)
        many_ones[..., 0] = 1
        output = rowan.rotate(many_ones, zero_vector)
        assert output.shape == shape_out

        # Two nonconforming array sizes
        with pytest.raises(ValueError):
            rowan.rotate(many_ones, np.repeat(zero_vector[np.newaxis, :], 2, axis=0))

        # Require broadcasting in multiple dimensions
        ones_quat = np.zeros((1, 1, 3, 8, 1, 4))
        ones_quat[..., 0] = 1
        zeros_vec = np.zeros((3, 5, 1, 1, 9, 3))
        shape = (3, 5, 3, 8, 9, 3)
        product = rowan.rotate(ones_quat, zeros_vec)
        assert product.shape == shape

        # Test complex rotations
        num_first = 8
        num_second = 5
        i1 = input1[:num_first, np.newaxis, :]
        i2 = vector_inputs[np.newaxis, :num_second, :]
        output = rowan.rotate(i1, i2)
        for i in range(num_first):
            for j in range(num_second):
                single_rot = rowan.rotate(i1[i, 0, :], i2[0, j, :])
                assert np.all(output[i, j, :] == single_rot)
