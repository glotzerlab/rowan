"""Test algorithms for point-cloud mapping."""
from __future__ import division, print_function, absolute_import

import unittest
import numpy as np

import rowan
from rowan import mapping, random, norm

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
            print("Testing size {}".format(num_points))

            points = np.random.rand(num_points, 3)
            rotation = random.rand(1)
            translation = np.random.rand(1, 3)

            transformed_points = rowan.rotate(rotation, points) + translation

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
                        rowan.rotate(q, points) + t
                        )
                    )

