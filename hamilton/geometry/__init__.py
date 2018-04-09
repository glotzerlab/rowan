# Copyright (c) 2018 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
"""This subpackage provides various tools for working with the geometric
representation of quaternions. A particular focus is computing the distance
between quaternions. These distance computations can be complicated,
particularly good metrics for distance on the Riemannian manifold representing
quaternions do not necessarily coincide with good metrics for similarities
between rotations.
"""

import numpy as np

from ..functions import norm

__all__ = []


def distance(p, q):
    """Determine the distance between quaternions p and q.

    This is the most basic distance that can be defined on
    the space of quaternions; it is the metric induced by
    the norm on this vector space
    :math:`\rho(p, q) = \lvert\lvert p - q \rvert\rvert`.

    Args:
        p ((...,4) np.array): First set of quaternions
        q ((...,4) np.array): Second set of quaternions

    Returns:
        An array containing the element-wise distances between
        the two sets of quaternions.

    Example::

        p = np.array([[1, 0, 0, 0]])
        q = np.array([[1, 0, 0, 0]])
        distance.distance(p, q)
    """
    return norm(p - q)


def sym_distance(p, q):
    """Determine the distance between quaternions p and q.

    This is a symmetrized version of :py:func:`distance` that
    accounts for the fact that :math:`p` and :math:`-p` represent
    identical rotations.

    Args:
        p ((...,4) np.array): First set of quaternions
        q ((...,4) np.array): Second set of quaternions

    Returns:
        An array containing the element-wise distances between
        the two sets of quaternions.

    Example::

        p = np.array([[1, 0, 0, 0]])
        q = np.array([[-1, 0, 0, 0]])
        distance.sym_distance(p, q) # 0
    """

    return np.minimum(norm(p - q), norm(p + q))
