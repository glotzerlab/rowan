# Copyright (c) 2018 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
R"""This subpackage provides the ability to compute the derivative and
integral of a quaternion.
"""
from __future__ import division, print_function, absolute_import

import numpy as np

from ..functions import multiply, _promote_vec, _validate_unit, exp

__all__ = ['derivative',
           'integrate']


def derivative(q, v):
    R"""Compute the instantaneous derivative of unit quaternions, which is
    defined as

    .. math::
        \dot{q} = \frac{1}{2} \boldsymbol{v} q

    A derivation is provided `here`_.
    For a more thorough explanation, see `this page`_.

    .. _here: http://web.cs.iastate.edu/~cs577/handouts/quaternion.pdf
    .. _this page: https://fgiesen.wordpress.com/
                   2012/08/24/quaternion-differentiation/

    Args:
        q ((...,4) np.array): Array of quaternions.
        v ((...,3) np.array): Array of angular velocities.

    Returns:
        Array of shape (..., 4) containing element-wise derivatives of q.

    Example::

        q_prime = rowan.calculus.derivative([1, 0, 0, 0], [1, 0, 0])
    """
    q = np.asarray(q)
    v = np.asarray(v)

    _validate_unit(q)
    return 0.5*multiply(q, _promote_vec(v))


def integrate(q, v, dt):
    R"""Integrate unit quaternions by angular velocity using the following
    equation:

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
        q ((...,4) np.array): Array of quaternions.
        v ((...,3) np.array): Array of angular velocities.
        dt ((...) np.array): Array of timesteps.

    Returns:
        Array of shape (..., 4) containing element-wise integrals of q.

    Example::

        v_next = rowan.calculus.integrate([1, 0, 0, 0], [0, 0, 1e-2], 1)
    """
    q = np.asarray(q)
    v = np.asarray(v)
    dt = np.asarray(dt)

    _validate_unit(q)

    return multiply(exp(_promote_vec(v*dt/2)), q)
