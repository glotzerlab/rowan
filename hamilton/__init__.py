# Copyright (c) 2018 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
"""Simple quaternion library containing standard methods"""

from __future__ import division, print_function, absolute_import

from ._functions import *

# Get the version
import os

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '_version.py')) as f:
    exec(f.read())

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
