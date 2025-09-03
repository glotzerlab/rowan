"""Test converting quaternions to and from Euler angles."""

import os
import textwrap
import unittest

import numpy as np
import pytest

import rowan

zero = np.array([0, 0, 0, 0])
one = np.array([1, 0, 0, 0])

TESTDATA_FILENAME = os.path.join(os.path.dirname(__file__), "files/test_arrays.npz")
with np.load(TESTDATA_FILENAME) as data:
    euler_angles = data["euler_angles"]
    euler_quaternions = data["euler_quats"]


class TestEuler(unittest.TestCase):
    """Test Euler angle conversions."""

    def test_from_euler(self):
        """Convert Euler angles to quaternions."""
        alpha, beta, gamma = [0, 0, 0]
        assert np.all(rowan.from_euler(alpha, beta, gamma) == np.array([1, 0, 0, 0]))

        alpha, beta, gamma = [np.pi / 2, np.pi / 2, 0]
        assert np.allclose(
            rowan.from_euler(alpha, beta, gamma, "zyx", "intrinsic"),
            np.array([0.5, -0.5, 0.5, 0.5]),
        )

        # Confirm broadcasting works from different Euler angles
        alpha, beta, gamma = [[0, np.pi / 2], [0, np.pi / 2], 0]
        assert np.allclose(
            rowan.from_euler(alpha, beta, gamma),
            np.array([[1, 0, 0, 0], [0.5, -0.5, 0.5, 0.5]]),
        )

        alpha = [[0, np.pi / 2], [0, np.pi / 2]]
        beta = [0, np.pi / 2]
        gamma = 0
        assert np.allclose(
            rowan.from_euler(alpha, beta, gamma),
            np.array(
                [
                    [[1, 0, 0, 0], [0.5, -0.5, 0.5, 0.5]],
                    [[1, 0, 0, 0], [0.5, -0.5, 0.5, 0.5]],
                ]
            ),
        )

        # More complicated test, checks 2d arrays and more complex angles
        alpha = euler_angles[:, 0]
        beta = euler_angles[:, 1]
        gamma = euler_angles[:, 2]
        assert np.allclose(
            rowan.from_euler(alpha, beta, gamma, "zyz", "intrinsic"), euler_quaternions
        )

        # Ensure proper errors are raised
        with pytest.raises(ValueError):
            rowan.from_euler(alpha, beta, gamma, "foo", "intrinsic")

        with pytest.raises(ValueError):
            rowan.from_euler(alpha, beta, gamma, "foo", "extrinsic")

        with pytest.raises(ValueError):
            rowan.from_euler(alpha, beta, gamma, "zyz", "bar")

    def test_to_euler(self):
        """Test conversion to Euler angles."""
        v = one
        assert np.all(rowan.to_euler(v) == np.array([0, 0, 0]))

        v = np.array([0.5, 0.5, 0.5, 0.5])
        assert np.all(rowan.to_euler(v) == np.array([np.pi / 2, 0, np.pi / 2]))

        # More complicated test, checks 2d arrays
        # and more complex angles
        assert np.allclose(
            rowan.to_euler(euler_quaternions, "zyz", "intrinsic"), euler_angles
        )

        # Ensure proper errors are raised
        with pytest.raises(ValueError):
            rowan.to_euler(euler_quaternions, "foo", "intrinsic")

        with pytest.raises(ValueError):
            rowan.to_euler(euler_quaternions, "foo", "extrinsic")

        with pytest.raises(ValueError):
            rowan.to_euler(euler_quaternions, "zyz", "bar")

        with pytest.raises(ValueError):
            rowan.to_euler(2 * one)

        with pytest.raises(ValueError):
            rowan.to_euler(zero)

    def test_from_to_euler(self):
        """2-way conversion starting from Euler angles."""
        np.random.seed(0)
        quats = rowan.normalize(np.random.rand(25, 4))
        conventions = [
            "xzx",
            "xyx",
            "yxy",
            "yzy",
            "zyz",
            "zxz",
            "xzy",
            "xyz",
            "yxz",
            "yzx",
            "zyx",
            "zxy",
        ]
        axis_types = ["extrinsic", "intrinsic"]

        for convention in conventions:
            for axis_type in axis_types:
                euler = rowan.to_euler(quats, convention, axis_type)
                out = rowan.from_euler(
                    euler[..., 0],
                    euler[..., 1],
                    euler[..., 2],
                    convention,
                    axis_type,
                )
                assert np.all(
                    np.logical_or(
                        np.isclose(out - quats, 0), np.isclose(out + quats, 0)
                    )
                ), f"Failed for convention {convention}, axis type {axis_type}"

    def test_to_from_euler(self):
        """2-way conversion starting from quaternions."""
        np.random.seed(0)
        angles_euler = np.pi * np.random.rand(100, 3)
        conventions_euler = ["xzx", "xyx", "yxy", "yzy", "zyz", "zxz"]

        # For Tait-Bryan angles the second angle must be between -pi/2 and pi/2
        angles_tb = angles_euler.copy()
        angles_tb[:, 1] -= np.pi / 2
        conventions_tb = ["xzy", "xyz", "yxz", "yzx", "zyx", "zxy"]

        axis_types = ["extrinsic", "intrinsic"]

        for convention in conventions_euler:
            for axis_type in axis_types:
                out = rowan.to_euler(
                    rowan.from_euler(
                        angles_euler[..., 0],
                        angles_euler[..., 1],
                        angles_euler[..., 2],
                        convention,
                        axis_type,
                    ),
                    convention,
                    axis_type,
                )
                assert np.all(
                    np.logical_or(
                        np.isclose(out - angles_euler, 0),
                        np.isclose(out + angles_euler, 0),
                    )
                ), f"Failed for convention {convention}, axis type {axis_type}"

        for convention in conventions_tb:
            for axis_type in axis_types:
                out = rowan.to_euler(
                    rowan.from_euler(
                        angles_tb[..., 0],
                        angles_tb[..., 1],
                        angles_tb[..., 2],
                        convention,
                        axis_type,
                    ),
                    convention,
                    axis_type,
                )
                assert np.all(
                    np.logical_or(
                        np.isclose(out - angles_tb, 0), np.isclose(out + angles_tb, 0)
                    )
                ), f"Failed for convention {convention}, axis type {axis_type}"

    def test_zero_beta(self):
        """Check cases where beta is 0."""
        # Since the Euler calculations are all done using matrices, it's easier
        # to construct the test cases by directly using matrices as well. We
        # assume gamma is 0 since, due to gimbal lock, only either alpha+gamma
        # or alpha-gamma is a relevant parameter, and we just scan the other
        # possible values. The actual function is defined such that gamma will
        # always be zero in those cases. We define the matrices using lambda
        # functions to support sweeping a range of values for alpha and beta,
        # specifically to test cases where signs flip e.g. cos(0) vs cos(pi).
        # These sign flips lead to changes in the rotation angles that must be
        # tested.
        mats_euler_intrinsic = [
            (
                "xzx",
                "intrinsic",
                lambda alpha, beta: [
                    [np.cos(beta), 0, 0],
                    [0, np.cos(beta) * np.cos(alpha), -np.sin(alpha)],
                    [0, np.cos(beta) * np.sin(alpha), np.cos(alpha)],
                ],
            ),
            (
                "xyx",
                "intrinsic",
                lambda alpha, beta: [
                    [np.cos(beta), 0, 0],
                    [0, np.cos(alpha), -np.cos(beta) * np.sin(alpha)],
                    [0, np.sin(alpha), np.cos(beta) * np.cos(alpha)],
                ],
            ),
            (
                "yxy",
                "intrinsic",
                lambda alpha, beta: [
                    [np.cos(alpha), 0, np.cos(beta) * np.sin(alpha)],
                    [0, np.cos(beta), 0],
                    [-np.sin(alpha), 0, np.cos(beta) * np.cos(alpha)],
                ],
            ),
            (
                "yzy",
                "intrinsic",
                lambda alpha, beta: [
                    [np.cos(beta) * np.cos(alpha), 0, np.sin(alpha)],
                    [0, np.cos(beta), 0],
                    [-np.cos(beta) * np.sin(alpha), 0, np.cos(alpha)],
                ],
            ),
            (
                "zyz",
                "intrinsic",
                lambda alpha, beta: [
                    [np.cos(beta) * np.cos(alpha), -np.sin(alpha), 0],
                    [np.cos(beta) * np.sin(alpha), np.cos(beta), 0],
                    [0, 0, np.cos(beta)],
                ],
            ),
            (
                "zxz",
                "intrinsic",
                lambda alpha, beta: [
                    [np.cos(alpha), -np.cos(beta) * np.sin(alpha), 0],
                    [np.sin(alpha), np.cos(beta) * np.cos(beta), 0],
                    [0, 0, np.cos(beta)],
                ],
            ),
        ]

        mats_tb_intrinsic = [
            (
                "xzy",
                "intrinsic",
                lambda alpha, beta: [
                    [0, -np.sin(beta), 0],
                    [np.sin(beta) * np.cos(alpha), 0, -np.sin(alpha)],
                    [np.sin(beta) * np.sin(alpha), 0, np.cos(alpha)],
                ],
            ),
            (
                "xyz",
                "intrinsic",
                lambda alpha, beta: [
                    [0, 0, np.sin(beta)],
                    [np.sin(beta) * np.sin(alpha), np.cos(alpha), 0],
                    [-np.sin(beta) * np.cos(alpha), np.sin(alpha), 0],
                ],
            ),
            (
                "yxz",
                "intrinsic",
                lambda alpha, beta: [
                    [np.cos(alpha), np.sin(beta) * np.sin(alpha), 0],
                    [0, 0, -np.sin(beta)],
                    [-np.sin(alpha), np.sin(beta) * np.cos(alpha), 0],
                ],
            ),
            (
                "yzx",
                "intrinsic",
                lambda alpha, beta: [
                    [0, -np.sin(beta) * np.cos(alpha), np.sin(alpha)],
                    [np.sin(beta), 0, 0],
                    [0, np.sin(beta) * np.sin(alpha), np.cos(alpha)],
                ],
            ),
            (
                "zyx",
                "intrinsic",
                lambda alpha, beta: [
                    [0, -np.sin(alpha), np.sin(beta) * np.cos(alpha)],
                    [0, np.cos(alpha), np.sin(beta) * np.sin(alpha)],
                    [-np.sin(beta), 0, 0],
                ],
            ),
            (
                "zxy",
                "intrinsic",
                lambda alpha, beta: [
                    [np.cos(alpha), 0, np.sin(beta) * np.sin(alpha)],
                    [np.sin(alpha), 0, -np.sin(beta) * np.cos(alpha)],
                    [0, -1, 0],
                ],
            ),
        ]

        # Extrinsic rotations can be tested identically to intrinsic rotations
        # in the case of proper Euler angles.
        mats_euler_extrinsic = [(m[0], "extrinsic", m[2]) for m in mats_euler_intrinsic]

        # For Tait-Bryan angles, extrinsic rotations axis order must be
        # reversed (since axes 1 and 3 are not identical), but more
        # importantly, due to the sum/difference of alpha and gamma that
        # arises, we need to test the negative of alpha to catch the dangerous
        # cases. In practice we get the same results since we're sweeping alpha
        # values in the tests below, but it's useful to set this up precisely.
        mats_tb_extrinsic = [
            (m[0][::-1], "extrinsic", lambda alpha, beta: m[2](-alpha, beta))  # noqa
            for m in mats_tb_intrinsic
        ]

        # Since angle representations may not be unique, checking that
        # quaternions are equal may not work. Instead we perform rotations and
        # check that they are identical.  For simplicity, we rotate the
        # simplest vector with all 3 components (otherwise tests won't catch
        # the problem because there's no component to rotate).
        test_vector = [1, 1, 1]

        mats_intrinsic = (mats_euler_intrinsic, mats_tb_intrinsic)
        mats_extrinsic = (mats_euler_extrinsic, mats_tb_extrinsic)

        # The beta angles are different for proper Euler angles and Tait-Bryan
        # angles because the relevant beta terms will be sines and cosines,
        # respectively.
        all_betas = ((0, np.pi), (np.pi / 2, -np.pi / 2))
        alphas = (0, np.pi / 2, np.pi, 3 * np.pi / 2)

        for mats in (mats_intrinsic, mats_extrinsic):
            for betas, mat_set in zip(all_betas, mats):
                for convention, axis_type, mat_func in mat_set:
                    quaternions = []
                    for beta in betas:
                        for alpha in alphas:
                            mat = mat_func(alpha, beta)
                            if np.linalg.det(mat) == -1:
                                # Some of these will be improper rotations.
                                continue
                            quat = rowan.from_matrix(mat)
                            quaternions.append(quat)
                            euler = rowan.to_euler(quat, convention, axis_type)
                            converted = rowan.from_euler(
                                *euler,
                                convention=convention,
                                axis_type=axis_type,
                            )
                            correct_rotation = rowan.rotate(quat, test_vector)
                            test_rotation = rowan.rotate(converted, test_vector)
                            assert np.allclose(
                                correct_rotation, test_rotation, atol=1e-06
                            ), textwrap.dedent("""\
                                Failed for convention {convention},
                                axis type {axis_type},
                                alpha = {alpha},
                                beta = {beta}.
                                Expected quaternion: {quat}.
                                Calculated: {converted}.
                                Expected vector: {correct_rotation}.
                                Calculated vector: {test_rotation}.""")

                    # For completeness, also test with broadcasting.
                    quaternions = np.asarray(quaternions).reshape(-1, 4)
                    all_euler = rowan.to_euler(quaternions, convention, axis_type)
                    converted = rowan.from_euler(
                        all_euler[..., 0],
                        all_euler[..., 1],
                        all_euler[..., 2],
                        convention,
                        axis_type,
                    )
                    assert np.allclose(
                        rowan.rotate(quaternions, test_vector),
                        rowan.rotate(converted, test_vector),
                        atol=1e-06,
                    )
