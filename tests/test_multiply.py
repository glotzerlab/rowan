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

    def test_broadcast(self):
        """Ensure broadcasting works"""
        # Multiply zeros, simple shape check
        shape = (45, 3, 13, 4)
        many_zeros = np.zeros(shape)
        product = quaternion.multiply(many_zeros, zero)
        self.assertTrue(product.shape == shape)

        # Two nonconforming array sizes
        with self.assertRaises(ValueError):
            quaternion.multiply(
                    many_zeros,
                    np.repeat(zero[np.newaxis, :], 2, axis=0)
                    )

        # Require broadcasting in multiple dimensions
        zeros_A = np.zeros((1, 1, 3, 8, 1, 4))
        zeros_B = np.zeros((3, 5, 1, 1, 9, 4))
        shape = (3, 5, 3, 8, 9, 4)
        product = quaternion.multiply(zeros_A, zeros_B)
        self.assertTrue(product.shape == shape)

        # Test some actual products
        num_first = 2
        num_second = 5
        i1 = input1[:num_first, np.newaxis, :]
        i2 = input1[np.newaxis, :num_second, :]
        product = quaternion.multiply(i1, i2)
        for i in range(num_first):
            for j in range(num_second):
                single_prod = quaternion.multiply(i1[i, 0, :], i2[0, j, :])
                self.assertTrue(np.all(product[i, j, :] == single_prod))
