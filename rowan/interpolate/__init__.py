# Copyright (c) 2019 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
"""Interpolate between pairs of quaternions.

The rowan package provides a simple interface to slerp, the standard method of
quaternion interpolation for two quaternions.
"""

import numpy as np

from ..functions import _validate_unit, conjugate, log, multiply, power

__all__ = ["slerp", "slerp_prime", "squad"]


def slerp(q0, q1, t, ensure_shortest=True):
    r"""Spherical linear interpolation between p and q.

    The `slerp formula <https://en.wikipedia.org/wiki/Slerp#Quaternion_Slerp>`_
    can be easily expressed in terms of the quaternion exponential (see
    :func:`rowan.exp`).

    Args:
        q0 ((..., 4) :class:`numpy.ndarray`):
            First array of quaternions.
        q1 ((..., 4) :class:`numpy.ndarray`):
            Second array of quaternions.
        t ((...) :class:`numpy.ndarray`):
            Interpolation parameter :math:`\in [0, 1]`
        ensure_shortest (bool):
            Flip quaternions to ensure we traverse the geodesic in the shorter
            (:math:`<180^{\circ}`) direction.

    .. note::

        Given inputs such that :math:`t\notin [0, 1]`, the values outside the
        range are simply assumed to be 0 or 1 (depending on which side of the
        interval they fall on).

    Returns:
        (..., 4) :class:`numpy.ndarray`: Interpolations between ``p`` and ``q``.

    Example::

        >>> import numpy as np
        >>> rowan.interpolate.slerp(
        ...     [[1, 0, 0, 0]], [[np.sqrt(2)/2, np.sqrt(2)/2, 0, 0]], 0.5)
        array([[0.92387953, 0.38268343, 0.        , 0.        ]])
    """
    _validate_unit(q0)
    _validate_unit(q1)
    t = np.clip(t, 0, 1)

    q0 = np.asarray(np.atleast_2d(q0))
    q1 = np.array(np.atleast_2d(q1))

    # Ensure that we turn the short way around
    if ensure_shortest:
        cos_theta = np.sum(q0 * q1, axis=-1)
        flip = cos_theta < 0
        q1[flip] *= -1

    return multiply(q0, power(multiply(conjugate(q0), q1), t))


def slerp_prime(q0, q1, t, ensure_shortest=True):
    r"""Compute the derivative of slerp.

    Args:
        q0 ((..., 4) :class:`numpy.ndarray`):
            First set of quaternions.
        q1 ((..., 4) :class:`numpy.ndarray`):
            Second set of quaternions.
        t ((...) :class:`numpy.ndarray`):
            Interpolation parameter :math:`\in [0, 1]`
        ensure_shortest (bool):
            Flip quaternions to ensure we traverse the geodesic in the shorter
            (:math:`<180^{\circ}`) direction

    Returns:
        (..., 4) :class:`numpy.ndarray`:
            The derivative of the interpolations between ``p`` and ``q``.

    Example::

        import numpy as np
        q_slerp_prime rowan.interpolate.slerp_prime(
            [[1, 0, 0, 0]], [[np.sqrt(2)/2, np.sqrt(2)/2, 0, 0]], 0.5)
    """
    _validate_unit(q0)
    _validate_unit(q1)
    t = np.clip(t, 0, 1)

    q0 = np.asarray(np.atleast_2d(q0))
    q1 = np.array(np.atleast_2d(q1))

    # Ensure that we turn the short way around
    if ensure_shortest:
        cos_theta = np.sum(q0 * q1, axis=-1)
        flip = cos_theta < 0
        q1[flip] *= -1

    return multiply(
        multiply(q0, power(multiply(conjugate(q0), q1), t)),
        log(multiply(conjugate(q0), q1)),
    )


def squad(p, a, b, q, t):
    r"""Cubically interpolate between p and q.

    The SQUAD formula is just a repeated application of Slerp between multiple
    quaternions as originally derived in [Shoemake85]_:

    .. math::
        \begin{equation}
            \textrm{squad}(p, a, b, q, t) = \textrm{slerp}(p, q, t)
            \left(\textrm{slerp}(p, q, t)^{-1}\textrm{slerp}(a, b, t)
            \right)^{2t(1-t)}
        \end{equation}

    .. [Shoemake85] Ken Shoemake. Animating rotation with quaternion curves.
        SIGGRAPH Comput. Graph., 19(3):245-254, July 1985.

    Args:
        p ((..., 4) :class:`numpy.ndarray`): First endpoint of interpolation.
        a ((..., 4) :class:`numpy.ndarray`): First control point of interpolation.
        b ((..., 4) :class:`numpy.ndarray`): Second control point of interpolation.
        q ((..., 4) :class:`numpy.ndarray`): Second endpoint of interpolation.
        t ((...) :class:`numpy.ndarray`): Interpolation parameter :math:`t \in [0, 1]`.

    Returns:
        (..., 4) :class:`numpy.ndarray`:
            Interpolations between ``p`` and ``q``.

    Example::

        >>> import numpy as np
        >>> rowan.interpolate.squad(
        ...     [1, 0, 0, 0], [np.sqrt(2)/2, np.sqrt(2)/2, 0, 0],
        ...     [0, np.sqrt(2)/2, np.sqrt(2)/2, 0],
        ...     [0, 0, np.sqrt(2)/2, np.sqrt(2)/2], 0.5)
        array([[0.64550177, 0.47254009, 0.52564058, 0.28937053]])
    """
    _validate_unit(p)
    _validate_unit(a)
    _validate_unit(b)
    _validate_unit(q)
    t = np.clip(t, 0, 1)

    return slerp(
        slerp(p, q, t, ensure_shortest=False),
        slerp(a, b, t, ensure_shortest=False),
        2 * t * (1 - t),
        ensure_shortest=False,
    )
