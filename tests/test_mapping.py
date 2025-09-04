"""Test algorithms for point-cloud mapping."""

import unittest
from itertools import product

import numpy as np
import scipy

from rowan import from_axis_angle, from_matrix, mapping, random, rotate

zero = np.array([0, 0, 0, 0])
one = np.array([1, 0, 0, 0])
half = np.array([0.5, 0.5, 0.5, 0.5])


def _cyclic_permutations(a):
    def make_permutation_generator(a, forward=True):
        """Return a list of functions that roll an array by 0,1,2...n."""
        n = len(a)
        sign = -1 if forward else 1
        return [
            lambda arr, k=j: arr[(sign * k) :] + arr[: (sign * k)] for j in range(n)
        ]

    return [op(a) for op in make_permutation_generator(a)]


class TestMapping(unittest.TestCase):
    """Test mapping functions."""

    def test_kabsch(self):
        """Perform a rotation and ensure that we can recover it."""
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
                assert np.logical_or(
                    np.allclose(rotation, q), np.allclose(rotation, -q)
                )
                assert np.allclose(translation, t)
            assert np.allclose(transformed_points, rotate(q, points) + t)

    def test_horn(self):
        """Perform a rotation and ensure that we can recover it."""
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
                assert np.logical_or(
                    np.allclose(rotation, q), np.allclose(rotation, -q)
                )
                assert np.allclose(translation, t)

            assert np.allclose(transformed_points, rotate(q, points) + t)

    def test_davenport(self):
        """Perform a rotation and ensure that we can recover it."""
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
                assert np.logical_or(
                    np.allclose(rotation, q), np.allclose(rotation, -q)
                )
                assert np.allclose(translation, t)
            assert np.allclose(transformed_points, rotate(q, points) + t)

    def test_procrustes(self):
        """Perform a rotation and ensure that we can recover it."""
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
                assert np.logical_or(
                    np.allclose(rotation, q), np.allclose(rotation, -q)
                )
                assert np.allclose(translation, t)
            assert np.allclose(transformed_points, rotate(q, points) + t)

    def test_equivalent(self):
        """Perform a rotation and ensure that we can recover it."""
        # Test on an octahedron
        points = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]

        # This is just a selected subset
        eq = [
            from_axis_angle([0, 0, 1], a) for a in [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        ]

        np.random.seed(0)
        rotation = random.rand(1)
        translation = np.random.rand(1, 3)

        transformed_points = rotate(rotation, points) + translation

        q, t = mapping.procrustes(points, transformed_points, equivalent_quaternions=eq)

        # Sort the points in a deterministic manner for comparison
        recovered_points = rotate(q, points) + t
        ordered_recovered_points = recovered_points[np.argsort(recovered_points[:, 0])]
        ordered_transformed_points = transformed_points[
            np.argsort(transformed_points[:, 0])
        ]

        assert np.allclose(ordered_recovered_points, ordered_transformed_points)

    def test_icp_exact(self):
        """Ensure that ICP is exact for corresponding inputs."""
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

            q, t, indices = mapping.icp(points, transformed_points, return_indices=True)
            q = from_matrix(q)

            # In the case of just two points, the mapping is not unique,
            # so we don't check the mapping itself, just the result.
            if i > 1:
                assert np.logical_or(
                    np.allclose(rotation, q), np.allclose(rotation, -q)
                )
                assert np.allclose(translation, t)
            assert np.allclose(transformed_points, rotate(q, points[indices]) + t)

    def test_icp_mismatched(self):
        """See how ICP works for non-corresponding inputs."""
        np.random.seed(0)

        # First test using unique matching, which should work.
        for i in range(2, 6):
            num_points = 2**i

            points = np.random.rand(num_points, 3)
            rotation = from_axis_angle([0.3, 0.3, 0.3], 0.3)
            translation = np.random.rand(1, 3)

            permutation = np.random.permutation(num_points)
            transformed_points = rotate(rotation, points[permutation]) + translation

            q, t, indices = mapping.icp(points, transformed_points, return_indices=True)
            q = from_matrix(q)

            deltas = transformed_points - (rotate(q, points[indices]) + t)
            norms = np.linalg.norm(deltas, axis=-1)
            # We have set some reasonable threshold for testing purposes, this is purely
            # a heuristic since we can't guarantee exact matches
            assert np.mean(norms) < 0.5

    def test_symmetric_quaternions_tetrahedral(self):
        """Verify our symmetrically equivalent quaternions are correct."""
        tet = [
            (-0.5, -0.5, 0.5),
            (-0.5, 0.5, -0.5),
            (0.5, -0.5, -0.5),
            (0.5, 0.5, 0.5),
        ]  # Verts in sorted order

        quats = mapping.SymmetricallyEquivalentQuaternions["T"]
        assert len(quats) == 24
        for quat in quats:
            res = sorted(rotate(quat, tet).tolist())
            np.testing.assert_array_equal(res, tet, err_msg=f"q={quat}")

        # Compare our array of generated quaternions to Scipy
        np.testing.assert_array_equal(
            sorted(quats.tolist()),
            sorted(
                [
                    *scipy.spatial.transform.Rotation.create_group("T")
                    .as_quat()
                    .tolist(),
                    *(
                        -scipy.spatial.transform.Rotation.create_group("T").as_quat()
                    ).tolist(),
                ]
            ),
        )

    def test_symmetric_quaternions_octahedral(self):
        """Verify our symmetrically equivalent quaternions are correct."""
        cube = [*product([-0.5, 0.5], repeat=3)]  # Verts in sorted order

        quats = mapping.SymmetricallyEquivalentQuaternions["O"]
        assert len(quats) == 48
        for quat in quats:
            res = sorted(rotate(quat, cube).tolist())
            np.testing.assert_allclose(res, cube, err_msg=f"q={quat}", atol=4e-16)

        # Compare our array of generated quaternions to Scipy
        np.testing.assert_allclose(
            sorted(quats.tolist()),
            sorted(
                [
                    *scipy.spatial.transform.Rotation.create_group("O")
                    .as_quat()
                    .tolist(),
                    *(
                        -scipy.spatial.transform.Rotation.create_group("O").as_quat()
                    ).tolist(),
                ]
            ),
            atol=4e-16,
        )

    def test_symmetric_quaternions_icosahedral(self):
        """Verify our symmetrically equivalent quaternions are correct."""
        # φ = (1 + np.sqrt(5)) / 2
        # icos = np.sort(
        #     [
        #         *_cyclic_permutations([0, -1, -φ]),
        #         *_cyclic_permutations([0, -1, φ]),
        #         *_cyclic_permutations([0, 1, -φ]),
        #         *_cyclic_permutations([0, 1, φ]),
        #     ],
        #     axis=0,
        # )

        quats = mapping.SymmetricallyEquivalentQuaternions["I"]
        assert len(quats) == 120

        # TODO: we get the correct vertices, but the ordering is difficult to ensure.
        # Is there a better way to test for equivalence here?
        # for i, quat in enumerate(quats):
        #     res = np.sort(rotate(quat, icos).round(14), axis=0)
        #     np.testing.assert_allclose(res, icos, err_msg=f"q{i}={quat}", atol=4e-16)

        # Compare our array of generated quaternions to Scipy
        np.testing.assert_allclose(
            np.sort(quats, axis=0),
            np.sort(
                [
                    *scipy.spatial.transform.Rotation.create_group("I").as_quat(),
                    *(-scipy.spatial.transform.Rotation.create_group("I").as_quat()),
                ],
                axis=0,
            ),
            atol=4e-16,
        )


if __name__ == "__main__":
    unittest.main()
