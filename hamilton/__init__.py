# Copyright (c) 2018 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
"""A library for quaternion operations"""

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
