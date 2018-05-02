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
    R"""Compute the instantaneous derivative of unit quaternions.

    Args:
        q ((...,4) np.array): Array of quaternions.
        v ((...,3) np.array): Array of angular velocities.

    Returns:
        Array of shape (..., 4) containing element-wise derivatives of q.
    """
    q = np.asarray(q)
    v = np.asarray(v)

    _validate_unit(q)
    return 0.5*multiply(q, _promote_vec(v))


def integrate(q, v, dt):
    R"""Integrate unit quaternions by angular velocity.

    Args:
        q ((...,4) np.array): Array of quaternions.
        v ((...,3) np.array): Array of angular velocities.
        dt ((...) np.array): Array of timesteps.

    Returns:
        Array of shape (..., 4) containing element-wise integrals of q.

    Example::
        q = np.array([1, 0, 0, 0])
        v = np.array([0, 0, 1e-2])
        v_next = integrate(q, v, 1)
    """
    q = np.asarray(q)
    v = np.asarray(v)
    dt = np.asarray(dt)

    _validate_unit(q)

    return multiply(exp(_promote_vec(v*dt/2)), q)
