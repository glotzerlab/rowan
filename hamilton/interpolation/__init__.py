# Copyright (c) 2018 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
R"""This subpackage contains tools for interpolating between quaternions.
In addition to the standard SLERP algorithm, there are also more advanced
methods for more precise interpolation and interpolating between multiple
quaternions.
"""

import numpy as np

from ..functions import norm, exp, multiply, inverse, log

__all__ = []


def slerp(p, q, t):
    R"""Perform SLERP interpolation between quaternions in p and q.

    Args:
        p ((...,4) np.array): First set of quaternions
        q ((...,4) np.array): Second set of quaternions
        t ((...,4) np.array): The SLERP parameter for how far to interpolate.

    Returns:
        An array containing the interpolated quaternions (broadcasted as
        needed).

    Example::

        TBD

    """
    return norm(p - q)

