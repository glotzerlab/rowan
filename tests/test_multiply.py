"""Test the multiplication of quaternions for various array sizes"""
from __future__ import division, print_function, absolute_import

import unittest
import numpy as np
import os

import hamilton as quaternion

zero = np.array([0, 0, 0, 0])
one = np.array([1, 0, 0, 0])

# Load test files
TESTDATA_FILENAME = os.path.join(
    os.path.dirname(__file__),
    'files/test_arrays.npz')
with np.load(TESTDATA_FILENAME) as data:
    input1 = data['input1']
    input2 = data['input2']
    stored_product = data['product']


class TestMultiply(unittest.TestCase):
    """Test the core multiplication operation"""

    def test_single_quaternion(self):
        """Simplest case of quaternion multiplication"""
        # Multiply zeros
        product = quaternion.multiply(zero, zero)
        self.assertTrue(np.all(product == np.array([0, 0, 0, 0])))

        # Multiply ones
        product = quaternion.multiply(one, one)
        self.assertTrue(np.all(product == np.array([1, 0, 0, 0])))

    def test_2d_array(self):
        """Multiplying arrays of quaternions"""
        zeros = np.repeat(zero[np.newaxis, :], 10, axis=0)
        ones = np.repeat(one[np.newaxis, :], 10, axis=0)

        # Multiply zeros
        product = quaternion.multiply(zeros, zeros)
        self.assertTrue(np.all(product == np.repeat(
            np.array([0, 0, 0, 0])[np.newaxis, :], 10, axis=0)))

        # Multiply ones
        product = quaternion.multiply(ones, ones)
        self.assertTrue(np.all(product == np.repeat(
            np.array([1, 0, 0, 0])[np.newaxis, :], 10, axis=0)))

        # Complex random array
        product = quaternion.multiply(input1, input2)
        self.assertTrue(np.allclose(product, stored_product))

    def test_3d_array(self):
        """Multiplying higher dimensional arrays of quaternions"""
        num_reps = 20
        expanded_shape = (int(num_reps / 5), 5, 4)
        zeros = np.reshape(
            np.repeat(zero[np.newaxis, :], num_reps, axis=0), expanded_shape)
        ones = np.reshape(
            np.repeat(one[np.newaxis, :], num_reps, axis=0), expanded_shape)
        expected_product_zeros = np.reshape(
            np.repeat(
                np.array([0, 0, 0, 0])[np.newaxis, :],
                num_reps,
                axis=0),
            expanded_shape)
        expected_product_ones = np.reshape(
            np.repeat(
                np.array([1, 0, 0, 0])[np.newaxis, :],
                num_reps,
                axis=0),
            expanded_shape)

        # Zeros
        product = quaternion.multiply(zeros, zeros)
        self.assertTrue(np.all(product == expected_product_zeros))

        # Ones
        product = quaternion.multiply(ones, ones)
        self.assertTrue(np.all(product == expected_product_ones))

        # Complex random array
        num_reps = input1.shape[0]
        expanded_shape = (int(num_reps / 5), 5, 4)
        product = quaternion.multiply(
            np.reshape(
                input1, expanded_shape), np.reshape(
                input2, expanded_shape))
        self.assertTrue(
            np.allclose(
                product,
                np.reshape(
                    stored_product,
                    expanded_shape)))
