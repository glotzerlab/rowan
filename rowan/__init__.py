# Copyright (c) 2018 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
R"""
The core :py:mod:`rowan` package contains functions for operating on
quaternions. The core package is focused on robust implementations of key
functions like multiplication, exponentiation, norms, and others. Simple
functionality such as addition is inherited directly from numpy due to
the representation of quaternions as numpy arrays. Many core numpy functions
implemented for normal arrays are reimplemented to work on quaternions (
such as :py:func:`allclose` and :py:func:`isfinite`). Additionally, `numpy
broadcasting
<https://docs.scipy.org/doc/numpy-1.14.0/user/basics.broadcasting.html#>`_
is enabled throughout rowan unless otherwise specified. This means that
any function of 2 (or more) quaternions can take arrays of shapes that do
not match and return results according to numpy's broadcasting rules.
"""

from __future__ import division, print_function, absolute_import

from .functions import (conjugate, multiply, norm, normalize, rotate,
                        vector_vector_rotation, from_euler, to_euler,
                        from_matrix, to_matrix, from_axis_angle, to_axis_angle,
                        from_mirror_plane, reflect, exp, log, log10, logb,
                        power, isnan, isinf, isfinite, equal, not_equal,
                        allclose, isclose, inverse, divide, expb, exp10)

# Get the version
import os

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '_version.py')) as f:
    exec(f.read())

__all__ = ['conjugate',
           'exp',
           'expb',
           'exp10',
           'log',
           'logb',
           'log10',
           'power',
           'multiply',
           'norm',
           'normalize',
           'from_mirror_plane',
           'reflect',
           'rotate',
           'vector_vector_rotation',
           'from_euler',
           'to_euler',
           'from_matrix',
           'to_matrix',
           'from_axis_angle',
           'to_axis_angle',
           'isnan',
           'isinf',
           'isfinite',
           'equal',
           'not_equal',
           'allclose',
           'isclose',
           'inverse',
           'divide'
           ]
