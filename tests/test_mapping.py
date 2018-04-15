"""Test algorithms for point-cloud mapping."""
from __future__ import division, print_function, absolute_import

import unittest
import numpy as np

from rowan import mapping, random, rotate

zero = np.array([0, 0, 0, 0])
one = np.array([1, 0, 0, 0])
half = np.array([0.5, 0.5, 0.5, 0.5])


class TestMapping(unittest.TestCase):
    """Test mapping functions"""

    def test_kabsch(self):
        """Perform a rotation and ensure that we can recover it"""
        np.random.seed(0)

        for i in range(1, 12):
            num_points = 2**i

            points = np.random.rand(num_points, 3)
            rotation = random.rand(1)
            translation = np.random.rand(1, 3)

            transformed_points = rotate(rotation, points) + translation

            q, t = mapping.kabsch(points, transformed_points)

            # In the case of just two points, the mapping is not unique,
            # so we don't check the mapping itself, just the result.
            if i > 1:
                self.assertTrue(
                    np.logical_or(
                        np.allclose(rotation, q),
                        np.allclose(rotation, -q),
                        )
                    )
                self.assertTrue(np.allclose(translation, t))
            self.assertTrue(
                    np.allclose(
                        transformed_points,
                        rotate(q, points) + t
                        )
                    )

    def test_horn(self):
        """Perform a rotation and ensure that we can recover it"""
        np.random.seed(0)

        for i in range(1, 12):
            num_points = 2**i

            points = np.random.rand(num_points, 3)
            rotation = random.rand(1)
            translation = np.random.rand(1, 3)

            transformed_points = rotate(rotation, points) + translation

            q, t = mapping.horn(points, transformed_points)

#            print("Applied translation = ", translation)
#            print("Translation: ", t)
#            print("Found rotation: ", q)
#            print("Original rotation: ", rotation)

            # In the case of just two points, the mapping is not unique,
            # so we don't check the mapping itself, just the result.
            if i > 1:
                self.assertTrue(
                    np.logical_or(
                        np.allclose(rotation, q),
                        np.allclose(rotation, -q),
                        )
                    )
                self.assertTrue(np.allclose(translation, t))
#            print("Original points: ", transformed_points)
#            print("New points: ", rotate(q, points) + t)
            self.assertTrue(
                    np.allclose(
                        transformed_points,
                        rotate(q, points) + t
                        )
                    )

    def test_davenport(self):
        """Perform a rotation and ensure that we can recover it"""
        np.random.seed(0)

        for i in range(1, 12):
            num_points = 2**i

            points = np.random.rand(num_points, 3)
            rotation = random.rand(1)
            translation = np.random.rand(1, 3)

            transformed_points = rotate(rotation, points) + translation
#            print("points: \n", points)
#            print("transformed points: \n", transformed_points)

            q, t = mapping.davenport(points, transformed_points)

#            print("Applied translation = ", translation)
#            print("Translation: ", t)
#            print("Found rotation: ", q)
#            print("Original rotation: ", rotation)
#            print("Original rotation: ", rowan.to_matrix(rotation))
#            assert 0

            # In the case of just two points, the mapping is not unique,
            # so we don't check the mapping itself, just the result.
            if i > 1:
                self.assertTrue(
                    np.logical_or(
                        np.allclose(rotation, q),
                        np.allclose(rotation, -q),
                        )
                    )
                self.assertTrue(np.allclose(translation, t))
#            print("Original points: ", transformed_points)
#            print("New points: ", rotate(q, points) + t)
            self.assertTrue(
                    np.allclose(
                        transformed_points,
                        rotate(q, points) + t
                        )
                    )
