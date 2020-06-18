"""Test converting quaternions to and from axis-angle representation"""
from __future__ import absolute_import, division, print_function

import unittest

import numpy as np

import rowan


class TestFromAxisAngle(unittest.TestCase):
    """Test converting from axis angle representation"""

    def test_single(self):
        """Test rotation about an axis"""
        v = np.array([1, 0, 0])
        theta = np.pi
        quats = rowan.from_axis_angle(v, theta)
        self.assertTrue(quats.shape[:-1] == v.shape[:-1])
        self.assertTrue(np.allclose(quats, np.array([0, 1, 0, 0])))

    def test_multiple_vectors(self):
        """Test multiple vectors against an angle"""
        v = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        theta = np.pi
        quats = rowan.from_axis_angle(v, theta)
        self.assertTrue(quats.shape[:-1] == v.shape[:-1])
        self.assertTrue(
            np.allclose(quats, np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
        )

    def test_multiple(self):
        """Test multiple vectors against multiple angles"""
        v = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        theta = np.array([np.pi, np.pi / 2, np.pi / 3])
        quats = rowan.from_axis_angle(v, theta)
        self.assertTrue(quats.shape[:-1] == v.shape[:-1])
        self.assertTrue(
            np.allclose(
                quats,
                np.array(
                    [
                        [0, 1, 0, 0],
                        [np.sqrt(2) / 2, 0, np.sqrt(2) / 2, 0],
                        [np.sqrt(3) / 2, 0, 0, 1 / 2],
                    ]
                ),
            )
        )

    def test_complex(self):
        """Test higher dimensions and broadcasting"""
        # Various ways of producing the same output
        expected_output = (
            np.array(
                [
                    [0, 1, 0, 0],
                    [np.sqrt(2) / 2, 0, np.sqrt(2) / 2, 0],
                    [np.sqrt(3) / 2, 0, 0, 1 / 2],
                ]
            )[np.newaxis, np.newaxis, ...]
            .repeat(2, axis=0)
            .repeat(2, axis=1)
        )

        # Matching array shapes (no broadcasing at all)
        v = (
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])[np.newaxis, np.newaxis, ...]
            .repeat(2, axis=0)
            .repeat(2, axis=1)
        )
        theta = (
            np.array([np.pi, np.pi / 2, np.pi / 3])[np.newaxis, np.newaxis, ...]
            .repeat(2, axis=0)
            .repeat(2, axis=1)
        )

        quats = rowan.from_axis_angle(v, theta)
        self.assertTrue(quats.shape[:-1] == v.shape[:-1])
        self.assertTrue(np.allclose(quats, expected_output))

        # Broadcasting in theta
        theta_reduced = theta[0, :, ...]
        quats = rowan.from_axis_angle(v, theta_reduced)
        self.assertTrue(quats.shape[:-1] == v.shape[:-1])
        self.assertTrue(np.allclose(quats, expected_output))

        # Broadcasting in v
        v_reduced = v[:, 0, ...]
        quats = rowan.from_axis_angle(v_reduced, theta)
        self.assertTrue(quats.shape[:-1] == v.shape[:-1])
        self.assertTrue(np.allclose(quats, expected_output))

        # Broadcasting in both
        quats = rowan.from_axis_angle(
            v_reduced[:, np.newaxis, ...], theta_reduced[np.newaxis, :, ...]
        )
        self.assertTrue(quats.shape[:-1] == v.shape[:-1])
        self.assertTrue(np.allclose(quats, expected_output))


class TestToAxisAngle(unittest.TestCase):
    """Test converting to axis angle representation"""

    def test_div_zero(self):
        q = (1, 0, 0, 0)
        axes, angles = rowan.to_axis_angle(q)
        self.assertTrue(np.allclose(axes, [0, 0, 0]))
        self.assertEqual(angles, 0)

    def test_to_axis_angle(self):
        q = (np.sqrt(2) / 2, np.sqrt(2) / 2, 0, 0)
        axes, angles = rowan.to_axis_angle(q)
        self.assertTrue(np.allclose(axes, np.array([1, 0, 0])))
        self.assertTrue(np.allclose(angles, np.pi / 2))

        q2 = np.stack((q, q), axis=0)
        axes, angles = rowan.to_axis_angle(q2)
        self.assertTrue(np.allclose(axes, np.array([[1, 0, 0], [1, 0, 0]])))
        self.assertTrue(np.allclose(angles, [np.pi / 2, np.pi / 2]))
