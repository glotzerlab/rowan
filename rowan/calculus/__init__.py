# Copyright (c) 2019 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
r"""Compute derivatives and integrals of quaternions."""

import numpy as np

from ..functions import _promote_vec, _validate_unit, exp, multiply

__all__ = ["derivative", "integrate"]


def derivative(q, v):
    r"""Compute the instantaneous derivative of unit quaternions.

    Derivatives of quaternions are defined by the equation:

    .. math::
        \dot{q} = \frac{1}{2} \boldsymbol{v} q

    A derivation is provided `here`_.
    For a more thorough explanation, see `this page`_.

    .. _here: http://web.cs.iastate.edu/~cs577/handouts/quaternion.pdf
    .. _this page: https://fgiesen.wordpress.com/
                   2012/08/24/quaternion-differentiation/

    Args:
        q ((..., 4) :class:`numpy.ndarray`): Array of quaternions.
        v ((..., 3) :class:`numpy.ndarray`): Array of angular velocities.

    Returns:
        (..., 4) :class:`numpy.ndarray`: Derivatives of ``q``.

    Example::

        >>> rowan.calculus.derivative([1, 0, 0, 0], [1, 0, 0])
        array([0. , 0.5, 0. , 0. ])
    """
    q = np.asarray(q)
    v = np.asarray(v)

    _validate_unit(q)
    return 0.5 * multiply(q, _promote_vec(v))


def integrate(q, v, dt):
    r"""Integrate unit quaternions by angular velocity.

    The integral uses the following equation:

    .. math::
        \dot{q} = \exp\left(\frac{1}{2} \boldsymbol{v} dt\right) q

    Note that this formula uses the `quaternion exponential`_, so the argument
    to the exponential (which appears to be a vector) is promoted to a
    quaternion with scalar part 0 before the exponential is taken.
    A concise derivation is provided in `this paper`_.
    This `webpage`_ contains a more thorough explanation.

    .. _quaternion exponential: https://en.wikipedia.org/wiki/
                                Quaternion#Exponential,_logarithm,_and_power
    .. _this paper: https://www.researchgate.net/publication/
                    260466470_Geometric_Integration_of_Quaternions
    .. _webpage: https://www.ashwinnarayan.com/post/
                 how-to-integrate-quaternions/

    Args:
        q ((..., 4) :class:`numpy.ndarray`): Array of quaternions.
        v ((..., 3) :class:`numpy.ndarray`): Array of angular velocities.
        dt ((...) :class:`numpy.ndarray`): Array of timesteps.

    Returns:
        (..., 4) :class:`numpy.ndarray`: Integrals of ``q``.

    Example::

        >>> rowan.calculus.integrate([1, 0, 0, 0], [0, 0, 1e-2], 1)
        array([0.9999875 , 0.        , 0.        , 0.00499998])
    """
    q = np.asarray(q)
    v = np.asarray(v)
    dt = np.asarray(dt)

    _validate_unit(q)

    return multiply(exp(_promote_vec(v * dt / 2)), q)
