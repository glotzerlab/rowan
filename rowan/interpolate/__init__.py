# Copyright (c) 2018 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
R"""
The rowan package provides a simple interface to slerp, the standard method
of quaternion interpolation for two quaternions.
"""

from ..functions import power, multiply, conjugate, _validate_unit, log

__all__ = []


def slerp(q0, q1, t):
    R"""Linearly interpolate between p and q.

    The `slerp formula <https://en.wikipedia.org/wiki/Slerp#Quaternion_Slerp>`_
    can be easily expressed in terms of the quaternion exponential (see
    :py:func:`rowan.exp`).

    Args:
        q0 ((...,4) np.array): First set of quaternions
        q1 ((...,4) np.array): Second set of quaternions
        t ((...) np.array): Interpolation parameter :math:`\in [0, 1]`

    Returns:
        An array containing the element-wise interpolations between p and q.

    Example::

        q0 = np.array([[1, 0, 0, 0]])
        q1 = np.array([[np.sqrt(2)/2, np.sqrt(2)/2, 0, 0]])
        interpolate.slerp(q0, q1, 0.5)
    """
    _validate_unit(q0)
    _validate_unit(q1)
    return multiply(q0, power(multiply(conjugate(q0), q1), t))


def slerp_prime(q0, q1, t):
    R"""Compute the derivative of slerp.

    Args:
        q0 ((...,4) np.array): First set of quaternions
        q1 ((...,4) np.array): Second set of quaternions
        t ((...) np.array): Interpolation parameter :math:`\in [0, 1]`

    Returns:
        An array containing the element-wise derivatives of interpolations
        between p and q.

    Example::

        q0 = np.array([[1, 0, 0, 0]])
        q1 = np.array([[np.sqrt(2)/2, np.sqrt(2)/2, 0, 0]])
        interpolate.slerp_prime(q0, q1, 0.5)
    """
    _validate_unit(q0)
    _validate_unit(q1)
    return multiply(
            multiply(q0, power(multiply(conjugate(q0), q1), t)),
            log(conjugate(q0), q1)
            )
