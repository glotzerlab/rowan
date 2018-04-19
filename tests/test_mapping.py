"""Test algorithms for point-cloud mapping."""
from __future__ import division, print_function, absolute_import

import unittest
import numpy as np

from rowan import mapping, random, rotate, from_matrix, from_axis_angle

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

            R, t = mapping.kabsch(points, transformed_points)
            q = from_matrix(R)

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

    def test_davenport(self):
        """Perform a rotation and ensure that we can recover it"""
        np.random.seed(0)

        for i in range(1, 12):
            num_points = 2**i

            points = np.random.rand(num_points, 3)
            rotation = random.rand(1)
            translation = np.random.rand(1, 3)

            transformed_points = rotate(rotation, points) + translation

            q, t = mapping.davenport(points, transformed_points)

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

    def test_procrustes(self):
        """Perform a rotation and ensure that we can recover it"""
        np.random.seed(0)

        for i in range(1, 12):
            num_points = 2**i

            points = np.random.rand(num_points, 3)
            rotation = random.rand(1)
            translation = np.random.rand(1, 3)

            transformed_points = rotate(rotation, points) + translation

            q, t = mapping.procrustes(points, transformed_points)

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

    def test_equivalent(self):
        """Perform a rotation and ensure that we can recover it"""
        # Test on an octahedron
        points = [[1, 0, 0],
                  [-1, 0, 0],
                  [0, 1, 0],
                  [0, -1, 0],
                  [0, 0, 1],
                  [0, 0, -1]]

        # This is just a selected subset
        equivalent_orientations = [
                from_axis_angle([0, 0, 1], a) for a in
                [0, 2*np.pi/3, 4*np.pi/3]]

        np.random.seed(0)
        rotation = random.rand(1)
        translation = np.random.rand(1, 3)

        transformed_points = rotate(rotation, points) + translation

        q, t = mapping.procrustes(points, transformed_points,
                equivalent_quaternions=equivalent_orientations)

        self.assertTrue(
                np.allclose(
                    transformed_points,
                    rotate(q, points) + t
                    )
                )

    def test_icp_exact(self):
        """Ensure that ICP is exact for corresponding inputs"""
        # Note that we do not bother to test the non-unique matching since we
        # know it provides very poor results.
        np.random.seed(0)

        # First test using unique matching, which should work
        for i in range(2, 6):
            num_points = 2**i

            points = np.random.rand(num_points, 3)
            rotation = from_axis_angle([0.3, 0.3, 0.3], 0.3)
            translation = np.random.rand(1, 3)

            transformed_points = rotate(rotation, points) + translation

            q, t = mapping.icp(points, transformed_points)
            q = from_matrix(q)

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

    def test_icp_mismatched(self):
        """See how ICP works for non-corresponding inputs. Have set some
        reasonable threshold for testing purposes."""
        np.random.seed(0)

        # First test using unique matching, which should work
        for i in range(2, 6):
            num_points = 2**i

            points = np.random.rand(num_points, 3)
            rotation = from_axis_angle([0.3, 0.3, 0.3], 0.3)
            translation = np.random.rand(1, 3)

            transformed_points = rotate(rotation, points) + translation
            indices = np.arange(num_points)
            np.random.shuffle(indices)
            shuffled_points = points[indices]

            q, t = mapping.icp(shuffled_points, transformed_points)
            q = from_matrix(q)

            deltas = transformed_points - (rotate(q, shuffled_points) + t)
            norms = np.linalg.norm(deltas, axis=-1)
            # This is purely a heuristic, since we can't guarantee exact matches
            self.assertTrue(np.mean(norms) < 1)
