# Copyright (c) 2018 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
R"""This subpackage provides various tools for working with the geometric
representation of quaternions. A particular focus is computing the distance
between quaternions. These distance computations can be complicated,
particularly good metrics for distance on the Riemannian manifold representing
quaternions do not necessarily coincide with good metrics for similarities
between rotations. An overview of distance measurements can be found in
 `this paper <https://link.springer.com/article/10.1007/s10851-009-0161-2>`_.
"""

import numpy as np

from ..functions import norm, exp, multiply, inverse, log, _validate_unit

__all__ = ['distance',
           'sym_distance',
           'riemann_exp_map',
           'riemann_log_map'
           'intrinsic_distance'
           'sym_intrinsic_distance',
           'angle']


def distance(p, q):
    R"""Determine the distance between quaternions p and q.

    This is the most basic distance that can be defined on
    the space of quaternions; it is the metric induced by
    the norm on this vector space
    :math:`\rho(p, q) = \lvert\lvert p - q \rvert\rvert`.

    When applied to unit quaternions, this function produces
    values in the range :math:`[0, 2]`.

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
    R"""Determine the distance between quaternions p and q.

    This is a symmetrized version of :py:func:`distance` that
    accounts for the fact that :math:`p` and :math:`-p` represent
    identical rotations. This makes it a useful measure of rotation
    similarity.

    Args:
        p ((...,4) np.array): First set of quaternions
        q ((...,4) np.array): Second set of quaternions

    When applied to unit quaternions, this function produces
    values in the range :math:`[0, \sqrt{2}]`.

    Returns:
        An array containing the element-wise distances between
        the two sets of quaternions.

    Example::

        p = np.array([[1, 0, 0, 0]])
        q = np.array([[-1, 0, 0, 0]])
        distance.sym_distance(p, q) # 0
    """
    return np.minimum(norm(p - q), norm(p + q))


def riemann_exp_map(p, v):
    R"""Compute the exponential map on the Riemannian manifold
    :math:`\mathbb{H}^*` of nonzero quaterions.

    The nonzero quaternions form a Lie algebra :math:`\mathbb{H}^*` that
    is also a Riemannian manifold. In general, given a point p on a
    Riemannian manifold :math:`\mathcal{M}` and an element of the tangent
    space at p :math:`v \in T_p\mathcal{M}`, the Riemannian exponential
    map is defined by the geodesic starting at :math:`p` and tracing out
    an arc of length :math:`v` in the direction of :math:`v`. The tangent
    space of the Riemannian manifold of nonzero quaternions is just the
    set of all quaternions. Since the quaternions are a Lie group, we can
    use the Lie exponential map to define the mapping from the Lie algebra
    of quaternions onto the Lie group of nonzero quaternions, providing a
    way to quantify the length of the path in quaternion space.

    As a result, we can define the exponential map as

    .. math::
        \begin{equation}
            \textnormal{Exp}_p(v) = p*\xp(v)
        \end{equation}

    Args:
        p ((...,4) np.array): Points on the manifold of quaternions
        v ((...,4) np.array): Tangent vectors to traverse

    Returns:
        The endpoint of the geodesic that starts from p and travels
        a distance :math:`\lvert\lvert v\rvert\rvert` in the direction
        of :math:`v`.
        """
    return multiply(p, exp(v))


def riemann_log_map(p, q):
    R"""Compute the log map on the Riemannian manifold :math:`\mathbb{H}^*` of
    nonzero quaterions.

    This function inverts :py:func:`riemann_exp_map`. See that function for more
    details. In brief, given two quaternions p and q, this method returns a
    third quaternion parameterizing the geodesic passing from p to q. It is
    therefore an important measure of the distance between the two input
    quaternions.

    Args:
        p ((...,4) np.array): Starting points (quaternions)
        q ((...,4) np.array): Endpoints (quaternions)

    Returns:
        Quaternions pointing from p to q with magnitudes equal to the length of
        the geodesics joining these quaternions.
    """
    return log(multiply(inverse(q), p))


def intrinsic_distance(p, q):
    R"""Compute the intrinsic distance between quaternions on the manifold of
    quaternions.

    The quaternion distance is determined as the length of the quaternion
    joining the two quaternions (see :py:func:`riemann_log_map`). Rather
    than computing this directly, however, as shown in [Huynh09]_ we can
    compute this distance using the following equivalence:

    ..math::
        \begin{equation}
            \lvert\lvert \log(p q^{-1}) \rvert\rvert =
            2*np.cos(\lvert\langle p, q \rangle\rvert)
        \end{equation}

    When applied to unit quaternions, this function produces
    values in the range :math:`[0, \pi]`.

    .. [Huynh09] Huynh DQ (2009) Metrics for 3D rotations: comparison and
        analysis. J Math Imaging Vis 35(2):155-164

    Args:
        p ((...,4) np.array): First set of quaternions.
        q ((...,4) np.array): Second set of quaternions.

    Returns:
        The element-wise intrinsic distance between p and q.
    """
    # TODO: Consider implementing the optimization
#    if not np.allclose(2*np.arccos(np.linalg.norm(np.inner(p, q))),
#            norm(riemann_log_map(p, q))):
#        raise ValueError("Huh?")
#    return 2*np.arccos(np.linalg.norm(np.inner(p, q))),
    return norm(riemann_log_map(p, q))


def sym_intrinsic_distance(p, q):
    R"""Compute the intrinsic distance between quaternions on the manifold of
    quaternions.

    This is a symmetrized version of :py:func:`intrinsic_distance` that
    accounts for the double cover SU(2)->SO(3), making it a more useful
    metric for rotation similarity.

    When applied to unit quaternions, this function produces
    values in the range :math:`[0, \frac{\pi}{2}]`.

    Args:
        p ((...,4) np.array): First set of quaternions.
        q ((...,4) np.array): Second set of quaternions.

    Returns:
        The element-wise intrinsic distance between p and q.
    """
    return np.where(norm(p - q) < norm(p + q),
                    norm(riemann_log_map(p, q)),
                    norm(riemann_log_map(p, -q))
                    )


def angle(p):
    R"""Compute the angle of rotation of a quaternion.

    Note that this is identical to
    ``intrinsic_distance(p, np.array([1, 0, 0, 0]))``.

    Args:
        p ((...,4) np.array): Quaternions.

    Returns:
        The element-wise angles traced out by these rotations.
    """

    # TODO: Make sure all the quaternions are rotations
    # where they need to be.

    _validate_unit(p)
    return norm(log(p))