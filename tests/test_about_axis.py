"""Test the about axis function"""

import unittest
import numpy as np

import hamilton as quaternion


class TestAboutAxis(unittest.TestCase):
    def test_about_axis(self):
        """Test rotation about an axis"""
        # One vector, one angle
        v = np.array([1, 0, 0])
        theta = np.pi
        quats = quaternion.about_axis(v, theta)
        self.assertTrue(quats.shape[:-1] == v.shape[:-1])
        self.assertTrue(np.allclose(quats, np.array([0, 1, 0, 0])))

        # Multiple vectors, one angle
        v = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        theta = np.pi
        quats = quaternion.about_axis(v, theta)
        self.assertTrue(quats.shape[:-1] == v.shape[:-1])
        self.assertTrue(np.allclose(quats,
                                    np.array([[0, 1, 0, 0],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1]]))
                        )

        # Multiple vectors, multiple angles, matching dimension
        v = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        theta = np.array([np.pi, np.pi / 2, np.pi / 3])
        quats = quaternion.about_axis(v, theta)
        self.assertTrue(quats.shape[:-1] == v.shape[:-1])
        self.assertTrue(np.allclose(quats, np.array([[0, 1, 0, 0], [np.sqrt(
            2) / 2, 0, np.sqrt(2) / 2, 0], [np.sqrt(3) / 2, 0, 0, 1 / 2]])))

        # Higher dimensions, matching dimension
        expected_output = np.array([[0, 1, 0, 0],
                                    [np.sqrt(2) / 2, 0, np.sqrt(2) / 2, 0],
                                    [np.sqrt(3) / 2, 0, 0, 1 / 2]])[
            np.newaxis, np.newaxis, ...].repeat(
            2, axis=0).repeat(2, axis=1)

        v = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])[
            np.newaxis, np.newaxis, ...].repeat(
            2, axis=0).repeat(
            2, axis=1)
        theta = np.array([np.pi, np.pi / 2, np.pi / 3])[
            np.newaxis, np.newaxis, ...].repeat(
            2, axis=0).repeat(
            2, axis=1)

        quats = quaternion.about_axis(v, theta)
        self.assertTrue(quats.shape[:-1] == v.shape[:-1])
        self.assertTrue(np.allclose(quats, expected_output))

        # Higher dimensions, requires broadcasting
        v = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])[
            np.newaxis, np.newaxis, ...].repeat(
            2, axis=0).repeat(
            2, axis=1)
        theta = np.array([np.pi, np.pi / 2, np.pi / 3])

        quats = quaternion.about_axis(v, theta)
        self.assertTrue(quats.shape[:-1] == v.shape[:-1])
        self.assertTrue(np.allclose(quats, expected_output))
