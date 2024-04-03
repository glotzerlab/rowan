"""Test function to rotate vector onto vector."""

import unittest

import numpy as np

import rowan


class TestVectorVector(unittest.TestCase):
    """Test rotation of a vector onto another vector."""

    def test_simple(self):
        """Test finding quaternion to rotate a vector onto another vector."""
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([0, 1, 0])
        vec3 = np.array([0, 0, 1])
        quat = rowan.vector_vector_rotation(vec1, vec2)
        assert np.allclose(quat, np.array([[0, np.sqrt(2) / 2, np.sqrt(2) / 2, 0]]))
        quat = rowan.vector_vector_rotation(vec1, vec3)
        assert np.allclose(quat, np.array([[0, np.sqrt(2) / 2, 0, np.sqrt(2) / 2]]))

    def test_ap(self):
        """Test finding quaternion to rotate antiparallel vectors onto each other."""
        # For this test, there are multiple quaternions that would effect the
        # correct rotation, so rather than checking for a specific one we check
        # that the appropriate rotation results from applying the quaternion
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([0, 1, 0])
        quat = rowan.vector_vector_rotation(vec1, vec2)
        assert np.allclose(
            rowan.rotate(quat, vec1), vec2 / np.linalg.norm(vec2, axis=-1)
        )

        vec1 = np.array([1, 0, 0])
        vec2 = np.array([[0, 1, 0], [2, 0, 0], [-2, 0, 0]])
        quat = rowan.vector_vector_rotation(vec1, vec2)
        assert np.allclose(
            rowan.rotate(quat, vec1),
            vec2 / np.linalg.norm(vec2, axis=-1)[:, np.newaxis],
        )

        vec1 = np.array([0, 1, 0])
        vec2 = np.array([[0, 0, 1], [0, 2, 0], [0, -2, 0]])
        quat = rowan.vector_vector_rotation(vec1, vec2)
        assert np.allclose(
            rowan.rotate(quat, vec1),
            vec2 / np.linalg.norm(vec2, axis=-1)[:, np.newaxis],
        )

    def test_broadcast(self):
        """Test broadcasting."""
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([0, 1, 0])
        vec3 = np.array([0, 0, 1])

        arr1 = np.stack((vec2, vec3), axis=0)

        output = np.array(
            [
                [0, np.sqrt(2) / 2, np.sqrt(2) / 2, 0],
                [0, np.sqrt(2) / 2, 0, np.sqrt(2) / 2],
            ],
        )

        # Test both directions of single array broadcasting
        quat = rowan.vector_vector_rotation(vec1, arr1)
        assert np.allclose(quat, output)

        quat = rowan.vector_vector_rotation(arr1, vec1)
        assert np.allclose(quat, output)

        # Matching sizes
        arr2 = np.stack((vec1, vec1), axis=0)
        quat = rowan.vector_vector_rotation(arr1, arr2)
        assert np.allclose(quat, output)

        # Proper broadcasting
        arr1 = np.stack((vec2, vec3), axis=0)[:, np.newaxis, ...]
        arr2 = np.stack((vec1, vec1), axis=0)[np.newaxis, ...]
        bcast_output = output[:, np.newaxis, ...].repeat(2, axis=1)

        quat = rowan.vector_vector_rotation(arr1, arr2)
        assert np.allclose(quat, bcast_output)
