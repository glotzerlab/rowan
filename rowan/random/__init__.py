# Copyright (c) 2019 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
"""Various functions for generating random sets of rotation quaternions.

Random quaternions in general can be generated by simply randomly sampling 4-vectors,
and they can be converted into rotation quaternions by normalizing them. This package is
strictly focused on generating uniform samples on :math:`SO(3)`, ensuring that rotations
are uniformly sampled rather than just the space of unit quaternions.
"""

# TODO: Try implementing randomness using this paper or one of the ones it
# cites:
#    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2896220/
#
#    other alternatives:
#    http://home.lu.lv/~sd20008/papers/essays/Random%20unitary%20[paper].pdf

import numpy as np

__all__ = ["rand", "random_sample"]


def rand(*args):
    r"""Generate random rotations uniformly distributed on a unit sphere.

    This is a convenience function *a la* ``np.random.rand``. If you want a
    function that takes a tuple as input, use :func:`random_sample` instead.

    Args:
        args (tuple): The shape of the array to generate.

    Return:
        :class:`numpy.ndarray`:
            Random quaternions of the shape provided with an additional axis of length
            4.

    Example::

        >>> rowan.random.rand(3, 3, 2) # doctest: +SKIP
    """
    if len(args) == 0:
        return random_sample()
    return random_sample(args)


def random_sample(size=None):
    r"""Generate random rotations uniformly.

    In general, sampling from the space of all quaternions will not generate
    uniform rotations. What we want is a distribution that accounts for the
    density of rotations, *i.e.*, a distribution that is uniform with respect
    to the appropriate measure. The algorithm used here is detailed in
    [Shoe92]_.

    .. [Shoe92] Shoemake, K.: Uniform random rotations. In: D. Kirk, editor,
        Graphics Gems III, pages 124-132. Academic, New York, 1992.

    Args:
        size (tuple): The shape of the array to generate.

    Return:
        :class:`numpy.ndarray`:
            Random quaternions of the shape provided with an additional axis of length
            4.

    Example::

        >>> rowan.random.random_sample((3, 3, 2)) # doctest: +SKIP
    """
    if size is None:
        size = (3,)
    else:
        size += (3,)

    u = np.random.random_sample(size)

    theta1 = 2 * np.pi * u[..., 1]
    s1 = np.sin(theta1)
    c1 = np.cos(theta1)
    theta2 = 2 * np.pi * u[..., 2]
    s2 = np.sin(theta2)
    c2 = np.cos(theta2)

    r1 = np.sqrt(1 - u[..., 0])
    r2 = np.sqrt(u[..., 0])

    quats = np.stack((s1 * r1, c1 * r1, s2 * r2, c2 * r2), axis=-1)
    return quats.squeeze()
