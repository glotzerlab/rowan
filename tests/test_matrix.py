"""Test converting quaternions to and from rotation matrices"""

import unittest
import numpy as np

import quaternion as quaternion

zero = np.array([0, 0, 0, 0])
one = np.array([1, 0, 0, 0])
half = np.array([0.5, 0.5, 0.5, 0.5])


class TestMatrix(unittest.TestCase):
    """Test rotation matrix conversions"""

    def test_from_matrix(self):

        self.assertTrue(np.all(
            quaternion.from_matrix(np.eye(3)) == one
        ))

        mat = np.array([[0, 0, 1],
                        [1, 0, 0],
                        [0, 1, 0]])
        self.assertTrue(np.allclose(
            quaternion.from_matrix(mat), half
        ))

        mat = np.array([[0, 1, 0],
                        [0, 0, -1],
                        [-1, 0, 0]])
        v = np.copy(half)
        v[3] *= -1
        self.assertTrue(np.allclose(
            quaternion.from_matrix(mat), v
        ))

    def test_to_matrix(self):
        v = np.copy(zero)
        with self.assertRaises(ZeroDivisionError):
            quaternion.to_matrix(v)

        v = np.copy(one)
        self.assertTrue(np.all(
            quaternion.to_matrix(v) == np.eye(3)
        ))

        v = np.copy(half)
        self.assertTrue(np.allclose(
            quaternion.to_matrix(v),
            np.array([[0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0]])
        ))

        v[3] *= -1
        self.assertTrue(np.allclose(
            quaternion.to_matrix(v),
            np.array([[0, 1, 0],
                      [0, 0, -1],
                      [-1, 0, 0]])
        ))
