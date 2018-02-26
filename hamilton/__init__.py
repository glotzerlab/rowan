# Copyright (c) 2018 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
"""Simple quaternion library containing standard methods"""

from ._functions import *

__all__ = ['conjugate',
           'multiply',
           'norm',
           'normalize',
           'rotate',
           'about_axis',
           'vector_vector_rotation',
           'from_euler',
           'to_euler',
           'from_matrix',
           'to_matrix']

__version__ = 0.0
