# Copyright (c) 2019 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
"""The rowan package for working with quaternions.

The core :mod:`rowan` package contains functions for operating on
quaternions. The core package is focused on robust implementations of key
functions like multiplication, exponentiation, norms, and others. Simple
functionality such as addition is inherited directly from NumPy due to
the representation of quaternions as NumPy arrays. Many core NumPy functions
implemented for normal arrays are reimplemented to work on quaternions (
such as :func:`allclose` and :func:`isfinite`). Additionally, `NumPy
broadcasting
<https://docs.scipy.org/doc/numpy-1.14.0/user/basics.broadcasting.html#>`_
is enabled throughout rowan unless otherwise specified. This means that
any function of 2 (or more) quaternions can take arrays of shapes that do
not match and return results according to NumPy's broadcasting rules.
"""

from . import calculus, geometry, interpolate, mapping, random
from .functions import (
    allclose,
    conjugate,
    divide,
    equal,
    exp,
    exp10,
    expb,
    from_axis_angle,
    from_euler,
    from_matrix,
    from_mirror_plane,
    inverse,
    is_unit,
    isclose,
    isfinite,
    isinf,
    isnan,
    log,
    log10,
    logb,
    multiply,
    norm,
    normalize,
    not_equal,
    power,
    reflect,
    rotate,
    to_axis_angle,
    to_euler,
    to_matrix,
    vector_vector_rotation,
)

# Get the version
__version__ = "1.3.1"

__all__ = [
    "calculus",
    "geometry",
    "interpolate",
    "mapping",
    "random",
    "allclose",
    "conjugate",
    "divide",
    "exp",
    "expb",
    "exp10",
    "equal",
    "from_axis_angle",
    "from_euler",
    "from_matrix",
    "from_mirror_plane",
    "inverse",
    "isclose",
    "isinf",
    "isfinite",
    "isnan",
    "is_unit",
    "log",
    "logb",
    "log10",
    "multiply",
    "norm",
    "normalize",
    "not_equal",
    "power",
    "reflect",
    "rotate",
    "to_axis_angle",
    "to_euler",
    "to_matrix",
    "vector_vector_rotation",
]
