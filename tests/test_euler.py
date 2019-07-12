"""Test converting quaternions to and from Euler angles"""
from __future__ import division, print_function, absolute_import

import unittest
import numpy as np
import os

import rowan

zero = np.array([0, 0, 0, 0])
one = np.array([1, 0, 0, 0])

TESTDATA_FILENAME = os.path.join(
    os.path.dirname(__file__),
    'files/test_arrays.npz')
with np.load(TESTDATA_FILENAME) as data:
    euler_angles = data['euler_angles']
    euler_quaternions = data['euler_quats']


class TestEuler(unittest.TestCase):
    """Test Euler angle conversions"""

    def test_from_euler(self):
        """Convert Euler angles to quaternions"""
        alpha, beta, gamma = [0, 0, 0]
        self.assertTrue(np.all(
            rowan.from_euler(alpha, beta, gamma) ==
            np.array([1, 0, 0, 0])
        ))

        alpha, beta, gamma = [np.pi / 2, np.pi / 2, 0]
        self.assertTrue(np.allclose(
            rowan.from_euler(alpha, beta, gamma,
                             'zyx', 'intrinsic'),
            np.array([0.5, -0.5, 0.5, 0.5])
        ))

        # Confirm broadcasting works from different Euler angles
        alpha, beta, gamma = [[0, np.pi / 2], [0, np.pi / 2], 0]
        self.assertTrue(np.allclose(
            rowan.from_euler(alpha, beta, gamma),
            np.array([[1, 0, 0, 0], [0.5, -0.5, 0.5, 0.5]])
        ))

        alpha = [[0, np.pi / 2], [0, np.pi / 2]]
        beta = [0, np.pi / 2]
        gamma = 0
        self.assertTrue(np.allclose(
            rowan.from_euler(alpha, beta, gamma),
            np.array([[[1, 0, 0, 0], [0.5, -0.5, 0.5, 0.5]],
                     [[1, 0, 0, 0], [0.5, -0.5, 0.5, 0.5]]])
        ))

        # More complicated test, checks 2d arrays and more complex angles
        alpha = euler_angles[:, 0]
        beta = euler_angles[:, 1]
        gamma = euler_angles[:, 2]
        self.assertTrue(
            np.allclose(
                rowan.from_euler(alpha, beta, gamma, 'zyz', 'intrinsic'),
                euler_quaternions
            ))

        # Ensure proper errors are raised
        with self.assertRaises(ValueError):
            rowan.from_euler(alpha, beta, gamma, 'foo', 'intrinsic')

        with self.assertRaises(ValueError):
            rowan.from_euler(alpha, beta, gamma, 'foo', 'extrinsic')

        with self.assertRaises(ValueError):
            rowan.from_euler(alpha, beta, gamma, 'zyz', 'bar')

    def test_to_euler(self):
        """Test conversion to Euler angles"""
        v = one
        self.assertTrue(np.all(
            rowan.to_euler(v) == np.array([0, 0, 0])
        ))

        v = np.array([0.5, 0.5, 0.5, 0.5])
        self.assertTrue(np.all(
            rowan.to_euler(v) == np.array([np.pi / 2, 0, np.pi / 2])
        ))

        # More complicated test, checks 2d arrays
        # and more complex angles
        self.assertTrue(
            np.allclose(
                rowan.to_euler(euler_quaternions, 'zyz', 'intrinsic'),
                euler_angles
            ))

        # Ensure proper errors are raised
        with self.assertRaises(ValueError):
            rowan.to_euler(euler_quaternions, 'foo', 'intrinsic')

        with self.assertRaises(ValueError):
            rowan.to_euler(euler_quaternions, 'foo', 'extrinsic')

        with self.assertRaises(ValueError):
            rowan.to_euler(euler_quaternions, 'zyz', 'bar')

        with self.assertRaises(ValueError):
            rowan.to_euler(2*one)

        with self.assertRaises(ValueError):
            rowan.to_euler(zero)

    def test_from_to_euler(self):
        """2-way conversion starting from Euler angles"""
        np.random.seed(0)
        quats = rowan.normalize(np.random.rand(25, 4))
        conventions = ['xzx', 'xyx', 'yxy', 'yzy', 'zyz', 'zxz',
                       'xzy', 'xyz', 'yxz', 'yzx', 'zyx', 'zxy']
        axis_types = ['extrinsic', 'intrinsic']

        for convention in conventions:
            for axis_type in axis_types:
                euler = rowan.to_euler(quats, convention, axis_type)
                out = rowan.from_euler(
                        euler[..., 0], euler[..., 1], euler[..., 2],
                        convention, axis_type
                )
                self.assertTrue(
                    np.all(
                        np.logical_or(
                            np.isclose(out - quats, 0),
                            np.isclose(out + quats, 0)
                        )
                    ),
                    msg="Failed for convention {}, axis type {}".format(
                        convention, axis_type))

    def test_to_from_euler(self):
        """2-way conversion starting from quaternions"""
        np.random.seed(0)
        angles_euler = np.pi*np.random.rand(100, 3)
        conventions_euler = ['xzx', 'xyx', 'yxy', 'yzy', 'zyz', 'zxz']

        # For Tait-Bryan angles the second angle must be between -pi/2 and pi/2
        angles_tb = angles_euler.copy()
        angles_tb[:, 1] -= np.pi/2
        conventions_tb = ['xzy', 'xyz', 'yxz', 'yzx', 'zyx', 'zxy']

        axis_types = ['extrinsic', 'intrinsic']

        for convention in conventions_euler:
            for axis_type in axis_types:
                out = rowan.to_euler(
                    rowan.from_euler(
                        angles_euler[..., 0], angles_euler[..., 1],
                        angles_euler[..., 2], convention, axis_type),
                    convention, axis_type
                )
                self.assertTrue(
                    np.all(
                        np.logical_or(
                            np.isclose(out - angles_euler, 0),
                            np.isclose(out + angles_euler, 0)
                        )
                    ),
                    msg="Failed for convention {}, axis type {}".format(
                        convention, axis_type))

        for convention in conventions_tb:
            for axis_type in axis_types:
                out = rowan.to_euler(
                    rowan.from_euler(
                        angles_tb[..., 0], angles_tb[..., 1],
                        angles_tb[..., 2], convention, axis_type),
                    convention, axis_type
                )
                self.assertTrue(
                    np.all(
                        np.logical_or(
                            np.isclose(out - angles_tb, 0),
                            np.isclose(out + angles_tb, 0)
                        )
                    ),
                    msg="Failed for convention {}, axis type {}".format(
                        convention, axis_type))

    def test_zero_beta(self):
        """Check cases where beta is 0."""
        # The simplest failure cases will just involve two equal nonzero
        # quaternion components, which when normalized will be sqrt(2). These
        # simple cases correspond to alpha=pi/2 and gamma=0 (and beta=0).
        root2 = np.sqrt(2)/2

        # These are the quaternions that will fail because the normally used
        # entries in the matrix are 0, so the arctan functions return
        # meaningless values.

        # First set is the quaternions that result when evaluating the nonzero
        # trig function (the function in beta that leads to a 1 or -1) at the
        # positive value. This means that if the matrix entry is -sin(beta),
        # this quaternion corresponding to choosing beta = pi/2, so that matrix
        # entry will be -1.

        # Start with where either np.sin(beta) > 0 or np.cos(beta) > 0,
        # whichever one is the one that's alone. Similarly, start with cases
        # where np.sin(alpha) > 0 or np.cos(alpha) > 0, whichever one is
        # relevant.

        # Since everything is done using matrices, it's easier to construct
        # everything using matrices as well. We assume gamma is 0 (gimbal lock)
        # and just scan the other possible values.
        mats_euler_intrinsic = [
                ('xzx', 'intrinsic', lambda alpha, beta: [[np.cos(beta), 0, 0],
                                     [0, np.cos(beta)*np.cos(alpha), -np.sin(alpha)],
                                     [0, np.cos(beta)*np.sin(alpha), np.cos(alpha)]]),
                ('xyx', 'intrinsic', lambda alpha, beta: [[np.cos(beta), 0, 0],
                                     [0, np.cos(alpha), -np.cos(beta)*np.sin(alpha)],
                                     [0, np.sin(alpha), np.cos(beta)*np.cos(alpha)]]),
                ('yxy', 'intrinsic', lambda alpha, beta: [[np.cos(alpha), 0, np.cos(beta)*np.sin(alpha)],
                                     [0, np.cos(beta), 0],
                                     [-np.sin(alpha), 0, np.cos(beta)*np.cos(alpha)]]),
                ('yzy', 'intrinsic', lambda alpha, beta: [[np.cos(beta)*np.cos(alpha), 0, np.sin(alpha)],
                                     [0, np.cos(beta), 0],
                                     [-np.cos(beta)*np.sin(alpha), 0, np.cos(alpha)]]),
                ('zyz', 'intrinsic', lambda alpha, beta: [[np.cos(beta)*np.cos(alpha), -np.sin(alpha), 0],
                                     [np.cos(beta)*np.sin(alpha), np.cos(beta), 0],
                                     [0, 0, np.cos(beta)]]),
                ('zxz', 'intrinsic', lambda alpha, beta: [[np.cos(alpha), -np.cos(beta)*np.sin(alpha), 0],
                                     [np.sin(alpha), np.cos(beta)*np.cos(beta), 0],
                                     [0, 0, np.cos(beta)]]),
        ]

        mats_tb_intrinsic = [
                ('xzy', 'intrinsic', lambda alpha, beta: [[0, -np.sin(beta), 0],
                                     [np.sin(beta)*np.cos(alpha), 0, -np.sin(alpha)],
                                     [np.sin(beta)*np.sin(alpha), 0, np.cos(alpha)]]),
                ('xyz', 'intrinsic', lambda alpha, beta: [[0, 0, np.sin(beta)],
                                     [np.sin(beta)*np.sin(alpha), np.cos(alpha), 0],
                                     [-np.sin(beta)*np.cos(alpha), np.sin(alpha), 0]]),
                ('yxz', 'intrinsic', lambda alpha, beta: [[np.cos(alpha), np.sin(beta)*np.sin(alpha), 0],
                                     [0, 0, -np.sin(beta)],
                                     [-np.sin(alpha), np.sin(beta)*np.cos(alpha), 0]]),
                ('yzx', 'intrinsic', lambda alpha, beta: [[0, -np.sin(beta)*np.cos(alpha), np.sin(alpha)],
                                     [np.sin(beta), 0, 0],
                                     [0, np.sin(beta)*np.sin(alpha), np.cos(alpha)]]),
                ('zyx', 'intrinsic', lambda alpha, beta: [[0, -np.sin(alpha), np.sin(beta)*np.cos(alpha)],
                                     [0, np.cos(alpha), np.sin(beta)*np.sin(alpha)],
                                     [-np.sin(beta), 0, 0]]),
                ('zxy', 'intrinsic', lambda alpha, beta: [[np.cos(alpha), 0, np.sin(beta)*np.sin(alpha)],
                                     [np.sin(alpha), 0, -np.sin(beta)*np.cos(alpha)],
                                     [0, -1, 0]]),
        ]

        mats_euler_extrinsic = [
            (m[0][::-1], 'extrinsic', m[2]) for m in mats_euler_intrinsic
        ]

        # Need to add the requisite negative sign for the tb angles
        mats_tb_extrinsic = [
            (m[0][::-1], 'extrinsic',
                lambda alpha, beta: m[2](-alpha, beta)) for m in mats_tb_intrinsic
        ]


        # intrinsic_bpos_apos_euler = [
            # ('xzx', 'intrinsic', (root2, root2, 0, 0)),
            # ('xyx', 'intrinsic', (root2, root2, 0, 0)),
            # ('yxy', 'intrinsic', (root2, 0, root2, 0)),
            # ('yzy', 'intrinsic', (root2, 0, root2, 0)),
            # ('zyz', 'intrinsic', (root2, 0, 0, root2)),
            # ('zxz', 'intrinsic', (root2, 0, 0, root2)),
        # ]
        # intrinsic_bpos_aneg_euler = [
            # ('xzx', 'intrinsic', (-root2, root2, 0, 0)),
            # ('xyx', 'intrinsic', (-root2, root2, 0, 0)),
            # ('yxy', 'intrinsic', (-root2, 0, root2, 0)),
            # ('yzy', 'intrinsic', (-root2, 0, root2, 0)),
            # ('zyz', 'intrinsic', (-root2, 0, 0, root2)),
            # ('zxz', 'intrinsic', (-root2, 0, 0, root2)),
        # ]
        # intrinsic_bpos_apos_tb = [
            # ('xzy', 'intrinsic', (-0.5, -0.5, 0.5, -0.5)),
            # # ('xzy', 'intrinsic', (root2, root2, 0, 0)),
            # # ('xyz', 'intrinsic', (root2, root2, 0, 0)),
            # # ('yxz', 'intrinsic', (0.5, 0.5, 0.5, -0.5)),
            # # ('yzx', 'intrinsic', (-0.5, -0.5, -0.5, -0.5)),
            # # ('zyx', 'intrinsic', (-0.5, 0.5, -0.5, -0.5)),
            # # ('zxy', 'intrinsic', (-0.5, -0.5, -0.5, -0.5)),
        # ]
        # intrinsic_bpos_aneg_tb = [
            # ('xzy', 'intrinsic', (-0.5, 0.5, -0.5, -0.5)),
            # # ('xzy', 'intrinsic', (root2, root2, 0, 0)),
            # # ('xyz', 'intrinsic', (root2, root2, 0, 0)),
            # # ('yxz', 'intrinsic', (0.5, 0.5, 0.5, -0.5)),
            # # ('yzx', 'intrinsic', (-0.5, -0.5, -0.5, -0.5)),
            # # ('zyx', 'intrinsic', (-0.5, 0.5, -0.5, -0.5)),
            # # ('zxy', 'intrinsic', (-0.5, -0.5, -0.5, -0.5)),
        # ]
        # test_quats_intrinsic_positive = intrinsic_bpos_alph_neg_euler + intrinsic_bpos_alph_pos_euler + intrinsic_bpos_alph_pos_tb + intrinsic_bpos_alph_neg_tb

        # # These quaternions evaluate at a beta where the sign is flipped.
        # test_quats_intrinsic_negative = [
            # # ('xyx', 'intrinsic', (0, 0, -root2, -root2)),
            # # ('xzy', 'intrinsic', (root2, 0, 0, root2)),
            # # ('xzy', 'intrinsic', (-0.5, -0.5,  0.5, -0.5)),
            # # ('xyz', 'intrinsic', (root2, 0, root2, 0)),
            # # ('xyz', 'intrinsic', (-0.5, -0.5, -0.5, -0.5)),
            # # ('yxz', 'intrinsic', (root2, root2, 0, 0)),
            # # ('yzx', 'intrinsic', (root2, 0, 0, root2)),
            # # ('zyx', 'intrinsic', (root2, 0, root2, 0)),
            # # ('zxy', 'intrinsic', (root2, root2, 0, 0)),
        # ]


        # test_quats_intrinsic = test_quats_intrinsic_positive + test_quats_intrinsic_negative

        # test_quats_extrinsic = [
            # (q[0][::-1], 'extrinsic', q[2]) for q in test_quats_intrinsic
        # ]

        # test_quats = test_quats_intrinsic# + test_quats_extrinsic
        # test_quats = ('yzx', 'extrinsic', (root2, root2, 0, 0)),


        # Since angle representations may not be unique, checking that
        # quaternions are equal may not work. Instead we perform rotations and
        # check that they are identical.  For simplicity, we rotate the
        # simplest vector with all 3 components (otherwise tests won't catch
        # the problem because there's no component to rotate).
        test_vector = [1, 1, 1]
        all_betas = ((0, np.pi), (np.pi/2, -np.pi/2))
        alphas = (0, np.pi/2, np.pi, 3*np.pi/2)

        mats_intrinsic = (mats_euler_intrinsic, mats_tb_intrinsic)
        mats_extrinsic = (mats_euler_extrinsic, mats_tb_extrinsic)
        for mats in (mats_intrinsic, mats_extrinsic):
            for betas, mat_set in zip(all_betas, mats):
                # print(betas, mat_set)
                # print()
                for beta in betas:
                    for alpha in alphas:
                        for convention, axis_type, mat_func in mat_set:
                            # if convention == 'zyx':
                                # continue
                            mat = mat_func(alpha, beta)
                            # print(alpha, beta, axis_type, convention, mat)
                            if np.linalg.det(mat) == -1:
                                # Some of these will be improper rotations.
                                continue
                            quat = rowan.from_matrix(mat)
                            # print("Running")
                            euler = rowan.to_euler(
                                    quat,
                                    convention, axis_type
                                    )
                            converted = rowan.from_euler(
                                *euler,
                                convention, axis_type
                            )
                            try:
                                self.assertTrue(
                                    np.allclose(
                                        rowan.rotate(quat, test_vector),
                                        rowan.rotate(converted, test_vector),
                                        atol=1e-6
                                    ),
                                    msg="\nFailed for convention {}\naxis type {}\nquat {}\nconverted = {}\nrotate1 = {}\nrotate2 = {}\nmat= {}\nalpha = {}\nbeta = {}\neuler = {}".format(
                                        convention, axis_type, quat, converted,
                                        rowan.rotate(quat, test_vector),
                                        rowan.rotate(converted, test_vector),
                                        mat, alpha, beta, euler))
                            except AssertionError as e:
                                raise
                                print(e)
