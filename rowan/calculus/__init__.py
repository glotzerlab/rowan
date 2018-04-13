# Copyright (c) 2018 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
R"""This subpackage provides the ability to compute the derivative and
integral of a quaternion.
"""

import numpy as np

from ..functions import norm

__all__ = ['derivative',
           'integrate']


def derivative(q, v):
    R"""Compute the instantaneous derivative of quaternions.

    Args:
        q ((...,4) np.array): Quaternions to integrate
        v ((...,3) np.array): Integration rates

    Returns:
        An array containing the element-wise derivatives.
    """
    return norm(p - q)


def integrate(q, v, dt):
    R"""Integrate quaternions by some velocity.

    Args:
        q ((...,4) np.array): Quaternions to integrate
        v ((...,3) np.array): Integration rates
        dt ((...) np.array): Timesteps

    Returns:
        An array containing the element-wise integral of the
        quaternions in q.
    """
    return np.minimum(norm(p - q), norm(p + q))
