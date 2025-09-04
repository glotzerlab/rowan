"""."""

from itertools import combinations, permutations, product
from typing import Iterable

import numpy as np
from more_itertools import distinct_permutations
from numpy.typing import ArrayLike

from ..functions import from_axis_angle


def _cyclic_permutations(a):
    def make_permutation_generator(a, forward=True):
        """Return a list of functions that roll an array by 0,1,2...n"""
        n = len(a)
        sign = -1 if forward else 1
        return [
            lambda arr, k=j: arr[(sign * k) :] + arr[: (sign * k)] for j in range(n)
        ]

    return [op(a) for op in make_permutation_generator(a)]


def sign_changes(even: bool = True, d: int = 3):
    """Get all even (or odd) sign changes of a vector in ℝ3."""
    all_changes = np.array([*product([1, -1], repeat=d)])
    if even == "all":
        return all_changes
    return all_changes[np.prod(all_changes, axis=1) == (1 if even else -1)]


def generate_tetrahedral_group():
    """Generates the 24 quaternions of the chiral tetrahedral group <T>."""
    quats = [
        # 8 (4+4) quaternions from permutationsof (±1, 0, 0, 0)
        (1, 0, 0, 0),
        (-1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, -1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, -1, 0),
        (0, 0, 0, 1),
        (0, 0, 0, -1),
        *product([-0.5, 0.5], repeat=4),  # 16 quaternions from 1/2 * (±1, ±1, ±1, ±1)
    ]
    return np.array(quats)


def generate_octahedral_group():
    """Generates the 48 quaternions of the octahedral group."""
    c = 1 / np.sqrt(2)

    return np.array(
        [
            # 24 elements of the tetrahedral group
            *generate_tetrahedral_group(),
            # 24 quaternions from distinct permutations of 1/√2 * (±1, ±1, 0, 0)
            *distinct_permutations([c, c, 0, 0]),
            *distinct_permutations([c, -c, 0, 0]),
            *distinct_permutations([-c, -c, 0, 0]),
        ]
    )


def generate_cyclic_group(n: int, axis: ArrayLike = (0, 0, 1)):
    """Generates the 2n quaternions of the cyclic group <C_n>.

    Args:
        n (int): The index of the cyclic group C_n. The order of the
                 resulting group will be 2n.
        axis (np.ndarray): The axis of rotation for the group.

    Returns:
        np.ndarray: A (2n, 4) array of quaternions.
    """
    # return np.array([from_axis_angle(axis, 2 * np.pi * k / n) for k in range(n)])
    return from_axis_angle(axis, np.linspace(0, 2 * np.pi, n, endpoint=False)).round(15)


# def generate_dicyclic_group(n: int, axis: ArrayLike = (0, 0, 1)):
#     """Generates the 4n quaternions of the dicyclic group <D_n>.

#     Args:
#         n (int): The index of the cyclic group D_n. The order of the
#                  resulting group will be 4n.
#         axis (np.ndarray): The axis of rotation for the group.

#     Returns:
#         np.ndarray: A (4n, 4) array of quaternions.
#     """
#     return np.concatenate(
#         (
#             # 2n quaternions from the cyclic group
#             generate_cyclic_group(n, axis),
#             #     hetas = np.linspace(0, np.pi, n, endpoint=False)
#             # rv = np.pi * np.vstack([np.zeros(n), np.cos(thetas), np.sin(thetas)]).T
#             # g2 = np.roll(rv, axis, axis=1)
#             # from_axis_angle(axis, np.linspace(0, np.pi, n, endpoint=False)) # TODO
#         )
#     )


def even_permutations(it: Iterable):
    """Return permutations containing an even number of transpositions from an iterable.

    Args:
        it (typing.Iterable): The iterable to permute.

    Returns:
        typing.Generator: An (N!/2, len(it)) generator of permutations.
    """
    return (
        p for p in permutations(it) if not sum(a > b for a, b in combinations(p, 2)) % 2
    )


def generate_icosahedral_group():
    """Generates the 48 quaternions of the octahedral group."""
    φ = (1 + np.sqrt(5)) / 2

    return np.array(
        [
            # 24 elements of the tetrahedral group
            *generate_tetrahedral_group(),
            # 96 quaternions from even permutations of 1/2 * (0, ±1, ±1/φ, ±φ)
            *(
                q
                for a, b, c in product([1 / 2, -1 / 2], repeat=3)
                for q in even_permutations([0, a, b / φ, c * φ])
            ),
        ]
    )


print(len(generate_icosahedral_group()))
# print(generate_binary_icosahedral_group())
# TODO: test against Scipy, and make sure objects are invariant
