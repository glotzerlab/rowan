"""Test the rotation of quaternions for various array sizes"""

import unittest
import numpy as np
import os

import quaternion as quaternion

zero = np.array([0, 0, 0, 0])
one = np.array([1, 0, 0, 0])

zero_vector = np.array([0, 0, 0])
one_vector = np.array([1, 0, 0])

# Load test files
TESTDATA_FILENAME = os.path.join(
    os.path.dirname(__file__),
    'files/test_arrays.npz')
with np.load(TESTDATA_FILENAME) as data:
    input1 = data['input1']
    input2 = data['input2']
    stored_rotation = data['rotated_vectors']
    vector_inputs = data['vector_inputs']


class TestRotate(unittest.TestCase):
    """Test the core rotation operation"""

    def test_single_quaternion(self):
        """Testing trivial rotations"""
        self.assertTrue(
            np.all(
                quaternion.rotate(
                    zero,
                    one_vector) == zero_vector))
        self.assertTrue(
            np.all(
                quaternion.rotate(
                    one,
                    one_vector) == one_vector))

    def test_2d_array(self):
        """Rotating sets of vectors by sets of quaternions"""
        zeros = np.repeat(zero[np.newaxis, :], 10, axis=0)
        ones = np.repeat(one[np.newaxis, :], 10, axis=0)
        zero_vectors = np.repeat(zero_vector[np.newaxis, :], 10, axis=0)
        one_vectors = np.repeat(one_vector[np.newaxis, :], 10, axis=0)

        # Simple tests
        self.assertTrue(
            np.all(
                quaternion.rotate(
                    zeros,
                    one_vectors) == zero_vectors))
        self.assertTrue(
            np.all(
                quaternion.rotate(
                    ones,
                    one_vectors) == one_vectors))

        # Complex random array
        self.assertTrue(
            np.allclose(
                quaternion.rotate(
                    input1,
                    vector_inputs),
                stored_rotation))

    def test_3d_array(self):
        """Rotating higher dimensional arrays of vectors
            by arrays of quaternions"""
        num_reps = 20
        expanded_shape = (int(num_reps / 5), 5, 4)
        expanded_shape_vec = (int(num_reps / 5), 5, 3)
        zeros = np.reshape(
            np.repeat(zero[np.newaxis, :], num_reps, axis=0), expanded_shape)
        ones = np.reshape(
            np.repeat(one[np.newaxis, :], num_reps, axis=0), expanded_shape)
        zero_vectors = np.reshape(np.repeat(
            zero_vector[np.newaxis, :], num_reps, axis=0), expanded_shape_vec)
        one_vectors = np.reshape(np.repeat(
            one_vector[np.newaxis, :], num_reps, axis=0), expanded_shape_vec)

        # Simple tests
        self.assertTrue(
            np.all(
                quaternion.rotate(
                    zeros,
                    one_vectors) == zero_vectors))
        self.assertTrue(
            np.all(
                quaternion.rotate(
                    ones,
                    one_vectors) == one_vectors))

        # Complex random array
        num_reps = input1.shape[0]
        expanded_shape = (int(num_reps / 5), 5, 4)
        expanded_shape_vec = (int(num_reps / 5), 5, 3)
        rotation_result = quaternion.rotate(
            np.reshape(
                input1, expanded_shape), np.reshape(
                vector_inputs, expanded_shape_vec))
        self.assertTrue(
            np.allclose(
                rotation_result,
                np.reshape(
                    stored_rotation,
                    expanded_shape_vec)))
