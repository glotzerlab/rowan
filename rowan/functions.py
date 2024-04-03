# Copyright (c) 2019 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
r"""Submodule containing all standard functions."""

import numpy as np


def exp(q):
    r"""Compute the natural exponential function :math:`e^q`.

    The exponential of a quaternion in terms of its scalar and vector parts
    :math:`q = a + \boldsymbol{v}` is defined by exponential power series:
    formula :math:`e^x = \sum_{k=0}^{\infty} \frac{x^k}{k!}` as follows:

    .. math::
        \begin{align}
            e^q &= e^{a+v} \\
                &= e^a \left(\sum_{k=0}^{\infty} \frac{v^k}{k!} \right) \\
                &= e^a \left(\cos \lvert \lvert \boldsymbol{v} \rvert \rvert +
                    \frac{\boldsymbol{v}}{\lvert \lvert \boldsymbol{v} \rvert
                    \rvert} \sin \lvert \lvert \boldsymbol{v} \rvert \rvert
                    \right)
        \end{align}

    Args:
        q ((..., 4) :class:`numpy.ndarray`): Array of quaternions.

    Returns:
        (..., 4) :class:`numpy.ndarray`: Exponentials of ``q``.

    Example::

        >>> rowan.exp([1, 0, 0, 0])
        array([2.71828183, 0.        , 0.        , 0.        ])
    """
    q = np.asarray(q)

    expo = np.empty(q.shape)
    norms = np.linalg.norm(q[..., 1:], axis=-1)
    e = np.exp(q[..., 0])
    expo[..., 0] = e * np.cos(norms)
    norm_zero = np.isclose(norms, 0)
    not_zero = np.logical_not(norm_zero)
    if np.any(not_zero):
        expo[not_zero, 1:] = (
            e[not_zero, np.newaxis]
            * (q[not_zero, 1:] / norms[not_zero, np.newaxis])
            * np.sin(norms)[not_zero, np.newaxis]
        )
        if np.any(norm_zero):
            expo[norm_zero, 1:] = 0
    else:
        expo[..., 1:] = 0

    return expo


def expb(q, b):
    r"""Compute the exponential function :math:`b^q`.

    We define the exponential of a quaternion to an arbitrary base relative
    to the exponential function :math:`e^q` using the change of base
    formula as follows:

    .. math::
        \begin{align}
            b^q &= y \\
            q &= \log_b y  = \frac{\ln y}{\ln b}\\
            y &= e^{q\ln b}
        \end{align}

    Args:
        q ((..., 4) :class:`numpy.ndarray`): Array of quaternions.
        b ((...) :class:`numpy.ndarray`): Scalars to use as bases.

    Returns:
        (..., 4) :class:`numpy.ndarray`: Exponentials of ``q``.

    Example::

        >>> rowan.expb([1, 0, 0, 0], 2)
        array([2., 0., 0., 0.])
    """
    q = np.asarray(q)
    return exp(q * np.log(b))


def exp10(q):
    r"""Compute the exponential function :math:`10^q`.

    Wrapper around :func:`expb`.

    Args:
        q ((..., 4) :class:`numpy.ndarray`): Array of quaternions.

    Returns:
        (..., 4) :class:`numpy.ndarray`: Exponentials of ``q``.

    Example::

        >>> rowan.exp10([1, 0, 0, 0])
        array([10.,  0.,  0.,  0.])
    """
    return expb(q, 10)


def log(q):
    r"""Compute the quaternion natural logarithm.

    The natural of a quaternion in terms of its scalar and vector parts
    :math:`q = a + \boldsymbol{v}` is defined by inverting the exponential
    formula (see :func:`exp`), and is defined by the formula
    :math:`\frac{x^k}{k!}` as follows:

    .. math::
        \begin{equation}
            \ln(q) = \ln\lvert\lvert q \rvert\rvert +
                    \frac{\boldsymbol{v}}{\lvert\lvert \boldsymbol{v}
                    \rvert\rvert} \arccos\left(\frac{a}{q}\right)
        \end{equation}

    Args:
        q ((..., 4) :class:`numpy.ndarray`): Array of quaternions.

    Returns:
        (..., 4) :class:`numpy.ndarray`: Logarithms of ``q``.

    Example::

        >>> rowan.log([1, 0, 0, 0])
        array([0., 0., 0., 0.])
    """
    q = np.asarray(q)
    log = np.empty(q.shape)

    # We need all the norms to avoid divide by zeros later.
    # Can also use these to minimize the amount of work done.
    q_norms = norm(q)
    q_norm_zero = np.isclose(q_norms, 0)
    q_not_zero = np.logical_not(q_norm_zero)
    v_norms = np.linalg.norm(q[..., 1:], axis=-1)
    v_norm_zero = np.isclose(v_norms, 0)
    v_not_zero = np.logical_not(v_norm_zero)

    if np.any(q_not_zero):
        if np.any(q_norm_zero):
            log[q_norm_zero, 0] = -np.inf
        log[q_not_zero, 0] = np.log(q_norms[q_not_zero])
    else:
        log[..., 0] = -np.inf

    if np.any(v_not_zero):
        prefactor = np.empty(q[v_not_zero, 1:].shape)
        prefactor = q[v_not_zero, 1:] / v_norms[v_not_zero, np.newaxis]

        inv_cos = np.empty(v_norms[v_not_zero].shape)
        inv_cos = np.arccos(q[v_not_zero, 0] / q_norms[v_not_zero])

        if np.any(v_norm_zero):
            log[v_norm_zero, 1:] = 0
        log[v_not_zero, 1:] = prefactor * inv_cos[..., np.newaxis]
    else:
        log[..., 1:] = 0

    return log


def logb(q, b):
    r"""Compute the quaternion logarithm to some base b.

    The quaternion logarithm for arbitrary bases is defined using the
    standard change of basis formula relative to the natural logarithm.

    .. math::
        \begin{align}
            \log_b q &= y \\
            q &= b^y \\
            \ln q &= y \ln b \\
            y &= \log_b q = \frac{\ln q}{\ln b}
        \end{align}

    Args:
        q ((..., 4) :class:`numpy.ndarray`): Array of quaternions.
        b ((...) :class:`numpy.ndarray`): Scalars to use as log bases.

    Returns:
        (..., 4) :class:`numpy.ndarray`: Logarithms of ``q``.

    Example::

        >>> rowan.logb([1, 0, 0, 0], 2)
        array([0., 0., 0., 0.])
    """
    q = np.asarray(q)
    return log(q) / np.log(b)


def log10(q):
    r"""Compute the quaternion logarithm base 10.

    Wrapper around :func:`logb`.

    Args:
        q ((..., 4) :class:`numpy.ndarray`): Array of quaternions.

    Returns:
        (..., 4) :class:`numpy.ndarray`: Logarithms of ``q``.

    Example::

        >>> rowan.log10([1, 0, 0, 0])
        array([0., 0., 0., 0.])
    """
    q = np.asarray(q)
    return logb(q, 10)


def power(q, n):
    r"""Compute the power of a quaternion :math:`q^n`.

    Quaternions raised to a scalar power are defined according to the polar
    decomposition angle :math:`\theta` and vector :math:`\hat{u}`:
    :math:`q^n = \lvert\lvert q \rvert\rvert^n \left( \cos(n\theta) + \hat{u}
    \sin(n\theta)\right)`. However, this can be computed
    more efficiently by noting that :math:`q^n = \exp(n \ln(q))`.

    Args:
        q ((..., 4) :class:`numpy.ndarray`): Array of quaternions.
        n ((...) :class:`numpy.ndarray`): Scalars to exponentiate quaternions with.

    Returns:
        (..., 4) :class:`numpy.ndarray`: Powers of ``q``.

    Example::

        >>> rowan.power([1, 0, 0, 0], 5)
        array([1., 0., 0., 0.])
    """
    # TODO: Write polar decomposition function #noqa
    q = np.asarray(q)

    newshape = np.broadcast(q[..., 0], n).shape
    q = np.broadcast_to(q, newshape + (4,))
    n = np.broadcast_to(n, newshape)

    # Note that we follow the convention that 0^0 = 1
    check = n == 0
    if np.any(check):
        powers = np.empty(newshape + (4,))
        powers[check] = np.array([1, 0, 0, 0])
        not_check = np.logical_not(check)
        if np.any(not_check):
            powers[not_check] = exp(n[not_check, np.newaxis] * log(q[not_check, :]))
    else:
        powers = exp(n[..., np.newaxis] * log(q))

    return powers


def conjugate(q):
    r"""Conjugates an array of quaternions.

    Args:
        q ((..., 4) :class:`numpy.ndarray`): Array of quaternions.

    Returns:
        (..., 4) :class:`numpy.ndarray`: Conjugates of ``q``.

    Example::

        >>> rowan.conjugate([0.5, 0.5, -0.5, 0.5])
        array([ 0.5, -0.5,  0.5, -0.5])
    """
    # Don't use asarray to avoid modifying in place
    conjugate = np.array(q)
    conjugate[..., 1:] *= -1
    return conjugate


def inverse(q):
    r"""Compute the inverse of an array of quaternions.

    Args:
        q ((..., 4) :class:`numpy.ndarray`): Array of quaternions.

    Returns:
        (..., 4) :class:`numpy.ndarray`: Inverses of ``q``.

    Example::

        >>> rowan.inverse([1, 0, 0, 0])
        array([ 1., -0., -0., -0.])
    """
    # Copy input so that we can safely modify in place, ensure float.
    inverses = np.array(q, dtype=float)

    normsq = norm(inverses) ** 2
    if np.any(normsq):
        inverses[..., 1:] *= -1
        # Would like to do this in place, but can't guarantee type safety
        inverses[normsq > 0] = inverses[normsq > 0] / normsq[normsq > 0, np.newaxis]

    return inverses


def multiply(qi, qj):
    r"""Multiplies two arrays of quaternions.

    Note that quaternion multiplication is generally non-commutative, so the
    first and second set of quaternions must be passed in the correct order.

    Args:
        qi ((..., 4) :class:`numpy.ndarray`): Array of left quaternions.
        qj ((..., 4) :class:`numpy.ndarray`): Array of right quaternions.

    Returns:
        (..., 4) :class:`numpy.ndarray`:
            Element-wise products of ``q`` (obeying broadcasting rules up to the last
            dimension of ``qi`` and ``qj``).

    Example::

        >>> rowan.multiply([1, 0, 0, 0], [2, 0, 0, 0])
        array([2., 0., 0., 0.])
    """
    qi = np.asarray(qi)
    qj = np.asarray(qj)

    output = np.empty(np.broadcast(qi, qj).shape)

    output[..., 0] = qi[..., 0] * qj[..., 0] - np.sum(
        qi[..., 1:] * qj[..., 1:],
        axis=-1,
    )
    output[..., 1:] = (
        qi[..., 0, np.newaxis] * qj[..., 1:]
        + qj[..., 0, np.newaxis] * qi[..., 1:]
        + np.cross(qi[..., 1:], qj[..., 1:])
    )
    return output


def divide(qi, qj):
    r"""Divides two arrays of quaternions.

    Division is non-commutative; this function returns
    :math:`q_i q_j^{-1}`.

    Args:
        qi ((..., 4) :class:`numpy.ndarray`): Dividend quaternions.
        qj ((..., 4) :class:`numpy.ndarray`): Divisor quaternions.

    Returns:
        (..., 4) :class:`numpy.ndarray`:
            Element-wise quotients of ``q`` (obeying broadcasting rules up to the last
            dimension of ``qi`` and ``qj``).

    Example::

        >>> rowan.divide([1, 0, 0, 0], [2, 0, 0, 0])
        array([0.5, 0. , 0. , 0. ])
    """
    return multiply(qi, inverse(qj))


def norm(q):
    r"""Compute the quaternion norm.

    Args:
        q ((..., 4) :class:`numpy.ndarray`): Array of quaternions.

    Returns:
        (...) :class:`numpy.ndarray`: Norms of ``q``.

    Example::

        >>> rowan.norm([10, 0, 0, 0])
        10.0
    """
    q = np.asarray(q)
    return np.linalg.norm(q, axis=-1)


def normalize(q):
    r"""Normalize quaternions.

    Args:
        q ((..., 4) :class:`numpy.ndarray`): Array of quaternions.

    Returns:
        (..., 4) :class:`numpy.ndarray`: Normalized versions of ``q``.

    Example::

        >>> rowan.normalize([10, 0, 0, 0])
        array([1., 0., 0., 0.])
    """
    q = np.asarray(q)
    norms = norm(q)
    return q / norms[..., np.newaxis]


def is_unit(q):
    """Check if all input quaternions have unit norm.

    Args:
        q ((..., 4) :class:`numpy.ndarray`): Array of quaternions.

    Returns:
        (...) :class:`numpy.ndarray` of bool:
            Whether or not all inputs are unit quaternions.

    Example::

        >>> rowan.is_unit([10, 0, 0, 0])
        False
    """
    return np.allclose(norm(q), 1)


def _validate_unit(q, msg="Arguments must be unit quaternions"):
    """Ensure that all quaternions in q have unit norm."""
    if not is_unit(q):
        raise ValueError(msg)


def from_mirror_plane(x, y, z):
    r"""Generate quaternions from mirror plane equations.

    Reflection quaternions can be constructed from the form
    :math:`(0, x, y, z)`, *i.e.* with zero real component. The vector
    :math:`(x, y, z)` is the normal to the mirror plane.

    Args:
        x ((...) :class:`numpy.ndarray`): First planar component.
        y ((...) :class:`numpy.ndarray`): Second planar component.
        z ((...) :class:`numpy.ndarray`): Third planar component.

    Returns:
        (..., 4) :class:`numpy.ndarray`:
            Quaternions reflecting about the input plane :math:`(x, y, z)`.

    Example::

        >>> rowan.from_mirror_plane(*(1, 2, 3))
        array([0., 1., 2., 3.])
    """
    x, y, z = np.broadcast_arrays(x, y, z)
    q = np.empty(x.shape + (4,))
    q[..., 0] = 0
    q[..., 1] = x
    q[..., 2] = y
    q[..., 3] = z

    return q


def _promote_vec(v):
    """Promote vectors to their quaternion representation."""
    return np.concatenate((np.zeros(v.shape[:-1] + (1,)), v), axis=-1)


def reflect(q, v):
    r"""Reflect a list of vectors by a corresponding set of quaternions.

    For help constructing a mirror plane, see :func:`from_mirror_plane`.

    Args:
        q ((..., 4) :class:`numpy.ndarray`): Array of quaternions.
        v ((..., 3) :class:`numpy.ndarray`): Array of vectors.

    Returns:
        (..., 3) :class:`numpy.ndarray`:
            The result of reflecting ``v`` using ``q``.

    Example::

        >>> rowan.reflect([1, 0, 0, 0], [1, 1, 1])
        array([1., 1., 1.])
    """
    q = np.asarray(q)
    v = np.asarray(v)

    # Convert vector to quaternion representation
    quat_v = _promote_vec(v)
    return multiply(q, multiply(quat_v, q))[..., 1:]


def rotate(q, v):
    r"""Rotate a list of vectors by a corresponding set of quaternions.

    Args:
        q ((..., 4) :class:`numpy.ndarray`): Array of quaternions.
        v ((..., 3) :class:`numpy.ndarray`): Array of vectors.

    Returns:
        (..., 3) :class:`numpy.ndarray`:
            The result of rotating ``v`` using ``q``.

    Example::

        >>> rowan.rotate([1, 0, 0, 0], [1, 1, 1])
        array([1., 1., 1.])
    """
    q = np.asarray(q)
    v = np.asarray(v)

    # Convert vector to quaternion representation
    quat_v = _promote_vec(v)
    return multiply(q, multiply(quat_v, conjugate(q)))[..., 1:]


def _normalize_vec(v):
    r"""Normalize vectors."""
    v = np.asarray(v)
    norms = np.linalg.norm(v, axis=-1)
    return v / norms[..., np.newaxis]


def _vector_bisector(v1, v2):
    r"""Find the vector bisecting two vectors.

    Args:
        v1 ((..., 3) :class:`numpy.ndarray`): First array of vectors.
        v2 ((..., 3) :class:`numpy.ndarray`): Second array of vectors.

    Returns:
        (..., 3) :class:`numpy.ndarray`: The vector bisectors.
    """
    # Since np.inner and np.dot require manipulating the shapes in ways that
    # might be expensive and may not play nicely with broadcasting, we perform
    # the dot product manually on the broadcasted arrays
    v1_norm, v2_norm = np.broadcast_arrays(_normalize_vec(v1), _normalize_vec(v2))
    ap = np.isclose(np.sum(v1_norm * v2_norm, axis=-1), -1)

    if np.any(ap):
        result = np.empty(v1_norm.shape)

        # Parallel vectors are fine, only antiparallel vectors cause problems
        not_ap = np.logical_not(ap)
        result[not_ap] = _normalize_vec(v1_norm[not_ap] + v2_norm[not_ap])

        # To use cross products to find the normal, we need to choose a unit
        # vector that is also not (anti)parallel to the original. Keep two
        # options available to avoid this case.
        one_vec = np.array([[1, 0, 0]])
        other_one_vec = np.array([[0, 1, 0]])
        cross_element = np.where(
            np.isclose(np.abs(np.dot(v1_norm[ap], one_vec.T)), 1),
            other_one_vec,
            one_vec,
        )
        result[ap] = np.cross(v1_norm[ap], cross_element)

        return result
    return _normalize_vec(v1_norm + v2_norm)


def vector_vector_rotation(v1, v2):
    r"""Find the quaternion to rotate one vector onto another.

    .. note::

        Vector-vector rotation is underspecified, with one degree of freedom
        possible in the resulting quaternion. This method chooses to rotate by
        :math:`\pi` around the vector bisecting v1 and v2.

    Args:
        v1 ((..., 3) :class:`numpy.ndarray`): Array of vectors to rotate.
        v2 ((..., 3) :class:`numpy.ndarray`): Array of vector to rotate onto.

    Returns:
        (..., 4) :class:`numpy.ndarray`: Quaternions that rotate ``v1`` onto ``v2``.

    Example::

        >>> rowan.vector_vector_rotation([1, 0, 0], [0, 1, 0])
        array([6.12323400e-17, 7.07106781e-01, 7.07106781e-01, 0.00000000e+00])
    """
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    return from_axis_angle(_vector_bisector(v1, v2), np.pi)


def from_euler(alpha, beta, gamma, convention="zyx", axis_type="intrinsic"):
    r"""Convert Euler angles to quaternions.

    For generality, the rotations are computed by composing a sequence of
    quaternions corresponding to axis-angle rotations. While more efficient
    implementations are possible, this method was chosen to prioritize
    flexibility since it works for essentially arbitrary Euler angles as
    long as intrinsic and extrinsic rotations are not intermixed.

    Args:
        alpha ((...) :class:`numpy.ndarray`):
            Array of :math:`\alpha` values in radians.
        beta ((...) :class:`numpy.ndarray`):
            Array of :math:`\beta` values in radians.
        gamma ((...) :class:`numpy.ndarray`):
            Array of :math:`\gamma` values in radians.
        convention (str):
            One of the 12 valid conventions xzx, xyx, yxy, yzy, zyz, zxz, xzy, xyz, yxz,
            yzx, zyx, zxy.
        axis_type (str):
            Whether to use extrinsic or intrinsic rotations.

    Returns:
        (..., 4) :class:`numpy.ndarray`: Quaternions corresponding to the input angles.

    Example::

        >>> rowan.from_euler(0.3, 0.5, 0.7)
        array([0.91262714, 0.29377717, 0.27944389, 0.05213241])
    """
    angles = np.broadcast_arrays(alpha, beta, gamma)

    convention = convention.lower()

    if len(convention) > 3 or (set(convention) - set("xyz")):
        raise ValueError(
            "All acceptable conventions must be 3 \
character strings composed only of x, y, and z",
        )

    basis_axes = {
        "x": np.array([1, 0, 0]),
        "y": np.array([0, 1, 0]),
        "z": np.array([0, 0, 1]),
    }
    # Temporary method to ensure shapes conform
    for ax, vec in basis_axes.items():
        basis_axes[ax] = np.broadcast_to(vec, angles[0].shape + (vec.shape[-1],))

    # Split by convention, the easiest
    rotations = []
    if axis_type == "extrinsic":
        # Loop over the axes and add each rotation
        for i, char in enumerate(convention):
            ax = basis_axes[char]
            rotations.append(from_axis_angle(ax, angles[i]))
    elif axis_type == "intrinsic":
        for i, char in enumerate(convention):
            ax = basis_axes[char]
            rotations.append(from_axis_angle(ax, angles[i]))
            # Rotate the bases as well
            for key, value in basis_axes.items():
                basis_axes[key] = rotate(rotations[-1], value)
    else:
        raise ValueError("Only valid axis_types are intrinsic and extrinsic")

    # Compose the total rotation
    final_rotation = np.broadcast_to(np.array([1, 0, 0, 0]), rotations[0].shape)
    for q in rotations:
        final_rotation = multiply(q, final_rotation)

    return final_rotation


def to_euler(q, convention="zyx", axis_type="intrinsic"):  # noqa: C901
    r"""Convert quaternions to Euler angles.

    Euler angles are returned in the sequence provided, so in, *e.g.*,
    the default case ('zyx'), the angles returned are for a rotation
    :math:`Z(\alpha) Y(\beta) X(\gamma)`.

    .. note::

        In all cases, the :math:`\alpha` and :math:`\gamma` angles are
        between :math:`\pm \pi`. For proper Euler angles, :math:`\beta`
        is between :math:`0` and :math:`\pi`. For Tait-Bryan
        angles, :math:`\beta` lies between :math:`\pm\pi/2`.

    For simplicity, quaternions are converted to matrices, which are
    then converted to their Euler angle representations. All equations
    for rotations are derived by considering compositions of the `three
    elemental rotations about the three Cartesian axes
    <https://en.wikipedia.org/wiki/Rotation_matrix#Basic_rotations>`_. A
    Mathematica notebook describing this process can be found in the
    `misc subdirectory of the repository
    <https://github.com/glotzerlab/rowan/blob/master/misc/Euler.nb>`__.

    Extrinsic rotations are represented by matrix multiplications in
    the proper order, so :math:`z-y-x` is represented by the
    multiplication :math:`XYZ` so that the system is rotated first
    about :math:`Z`, then about :math:`Y`, then finally :math:`X`.
    For intrinsic rotations, the order of rotations is reversed,
    meaning that it matches the order in which the matrices actually
    appear *i.e.* the :math:`z-y'-x''` convention (yaw, pitch, roll)
    corresponds to the multiplication of matrices :math:`ZYX`.
    For proof of the relationship between intrinsic and extrinsic
    rotations, see the `Wikipedia page on Davenport chained rotations
    <https://en.wikipedia.org/wiki/Davenport_chained_rotations>`_.

    For more information, see the Wikipedia page for
    `Euler angles <https://en.wikipedia.org/wiki/Euler_angles>`_
    (specifically the section on converting between representations).

    .. warning::

        Euler angles are a highly problematic representation for a number of
        reasons, not least of which is the large number of possible conventions
        and their relative imprecision when compared to using quaternions (or
        axis-angle representations). If possible, you should avoid Euler angles
        and work with quaternions instead. If Euler angles are required, note
        that they are susceptible to `gimbal lock
        <https://en.wikipedia.org/wiki/Gimbal_lock>`_, which leads to ambiguity
        in the representation of a given rotation. To address this issue, in
        cases where gimbal lock arises, :func:`~.to_euler` adopts the
        convention that :math:`\gamma=0` and represents the rotation entirely
        in terms of :math:`\beta` and :math:`\alpha`.


    Args:
        q ((..., 4) :class:`numpy.ndarray`):
            Quaternions to transform.
        convention (str):
            One of the 6 valid conventions zxz, xyx, yzy, zyz, xzx, yxy.
        axis_type (str):
            Whether to use extrinsic or intrinsic.

    Returns:
        (..., 3) :class:`numpy.ndarray`:
            Euler angles :math:`(\alpha, \beta, \gamma)` corresponding to ``q``.

    Example::

        >>> import numpy as np
        >>> rands = np.random.rand(100, 3)
        >>> alpha, beta, gamma = rands.T
        >>> ql = rowan.from_euler(alpha, beta, gamma)
        >>> alpha_return, beta_return, gamma_return = np.split(
        ...     rowan.to_euler(ql), 3, axis = 1)
        >>> assert(np.allclose(alpha_return.flatten(), alpha))
        >>> assert(np.allclose(beta_return.flatten(), beta))
        >>> assert(np.allclose(gamma_return.flatten(), gamma))
    """
    q = np.asarray(q)
    _validate_unit(q)
    atol = 1e-3

    try:
        # Due to minor numerical imprecision, the to_matrix function could
        # generate a (very slightly) nonorthogonal matrix (e.g. with a norm of
        # 1 + 2e-8). That is sufficient to throw off the trigonometric
        # functions, so it's worthwhile to explicitly clip for safety,
        # especially since we've already checked the quaternion norm.
        mats = np.clip(to_matrix(q), -1, 1)
    except ValueError:
        raise ValueError("Not all quaternions in q are unit quaternions.")

    # For intrinsic angles, the matrix must be constructed in reverse order
    # e.g. Z(\alpha)Y'(\beta)Z''(\gamma) (where primes denote the rotated
    # frames) becomes the extrinsic rotation Z(\gamma)Y(\beta)Z(\alpha). Simply
    # for easier readability of order, matrices are constructed for the
    # intrinsic angle ordering and just reversed for extrinsic.
    if axis_type == "extrinsic":
        convention = convention[::-1]
    elif axis_type != "intrinsic":
        raise ValueError("The axis type must be either extrinsic or intrinsic")

    # We have to hardcode the different convention possibilities since they all
    # result in different matrices according to the rotation order. In all
    # possible compositions, there are cases where, given some 0 elements in
    # the matrix, the simplest combination of matrix elements will give the
    # wrong solution.  In those cases, we have to use other parts of the
    # matrix. In those cases, we have to be much more careful about signs,
    # because there are multiple places where negatives can come into play. Due
    # to gimbal lock, the alpha and gamma angles are no longer independent in
    # that case. By convention, we set gamma to 0 and solve for alpha in those
    # cases.

    # Classical Euler angles
    if convention == "xzx":
        beta = np.arccos(mats[..., 0, 0])
        multiplier = mats[..., 0, 0] if axis_type == "extrinsic" else 1
        where_zero = np.isclose(np.sin(beta), 0, atol=atol)

        gamma = np.where(where_zero, 0, np.arctan2(mats[..., 0, 2], -mats[..., 0, 1]))
        alpha = np.where(where_zero, 0, np.arctan2(mats[..., 2, 0], mats[..., 1, 0]))
        zero_terms = np.arctan2(-multiplier * mats[..., 1, 2], mats[..., 2, 2])
    elif convention == "xyx":
        beta = np.arccos(mats[..., 0, 0])
        multiplier = mats[..., 0, 0] if axis_type == "extrinsic" else 1
        where_zero = np.isclose(np.sin(beta), 0, atol=atol)

        gamma = np.where(where_zero, 0, np.arctan2(mats[..., 0, 1], mats[..., 0, 2]))
        alpha = np.where(where_zero, 0, np.arctan2(mats[..., 1, 0], -mats[..., 2, 0]))
        zero_terms = np.arctan2(multiplier * mats[..., 2, 1], mats[..., 1, 1])
    elif convention == "yxy":
        beta = np.arccos(mats[..., 1, 1])
        multiplier = mats[..., 1, 1] if axis_type == "extrinsic" else 1
        where_zero = np.isclose(np.sin(beta), 0, atol=atol)

        gamma = np.where(where_zero, 0, np.arctan2(mats[..., 1, 0], -mats[..., 1, 2]))
        alpha = np.where(where_zero, 0, np.arctan2(mats[..., 0, 1], mats[..., 2, 1]))
        zero_terms = np.arctan2(-multiplier * mats[..., 2, 0], mats[..., 0, 0])
    elif convention == "yzy":
        beta = np.arccos(mats[..., 1, 1])
        multiplier = mats[..., 1, 1] if axis_type == "extrinsic" else 1
        where_zero = np.isclose(np.sin(beta), 0, atol=atol)

        gamma = np.where(where_zero, 0, np.arctan2(mats[..., 1, 2], mats[..., 1, 0]))
        alpha = np.where(where_zero, 0, np.arctan2(mats[..., 2, 1], -mats[..., 0, 1]))
        zero_terms = np.arctan2(multiplier * mats[..., 0, 2], mats[..., 2, 2])
    elif convention == "zyz":
        beta = np.arccos(mats[..., 2, 2])
        multiplier = mats[..., 2, 2] if axis_type == "extrinsic" else 1
        where_zero = np.isclose(np.sin(beta), 0, atol=atol)

        gamma = np.where(where_zero, 0, np.arctan2(mats[..., 2, 1], -mats[..., 2, 0]))
        alpha = np.where(where_zero, 0, np.arctan2(mats[..., 1, 2], mats[..., 0, 2]))
        zero_terms = np.arctan2(-multiplier * mats[..., 0, 1], mats[..., 1, 1])
    elif convention == "zxz":
        beta = np.arccos(mats[..., 2, 2])
        multiplier = mats[..., 2, 2] if axis_type == "extrinsic" else 1
        where_zero = np.isclose(np.sin(beta), 0, atol=atol)

        gamma = np.where(where_zero, 0, np.arctan2(mats[..., 2, 0], mats[..., 2, 1]))
        alpha = np.where(where_zero, 0, np.arctan2(mats[..., 0, 2], -mats[..., 1, 2]))
        zero_terms = np.arctan2(multiplier * mats[..., 1, 0], mats[..., 0, 0])
    # Tait-Bryan angles
    elif convention == "xzy":
        beta = np.arcsin(-mats[..., 0, 1])
        where_zero = np.isclose(np.cos(beta), 0, atol=atol)

        gamma = np.where(where_zero, 0, np.arctan2(mats[..., 0, 2], mats[..., 0, 0]))
        alpha = np.where(where_zero, 0, np.arctan2(mats[..., 2, 1], mats[..., 1, 1]))
        zero_terms = np.arctan2(-mats[..., 1, 2], mats[..., 2, 2])
    elif convention == "xyz":
        beta = np.arcsin(mats[..., 0, 2])
        multiplier = mats[..., 0, 2] if axis_type == "extrinsic" else 1
        where_zero = np.isclose(np.cos(beta), 0, atol=atol)

        gamma = np.where(where_zero, 0, np.arctan2(-mats[..., 0, 1], mats[..., 0, 0]))
        alpha = np.where(where_zero, 0, np.arctan2(-mats[..., 1, 2], mats[..., 2, 2]))
        zero_terms = np.arctan2(multiplier * mats[..., 2, 1], mats[..., 1, 1])
    elif convention == "yxz":
        beta = np.arcsin(-mats[..., 1, 2])
        multiplier = mats[..., 1, 2] if axis_type == "extrinsic" else 1
        where_zero = np.isclose(np.cos(beta), 0, atol=atol)

        gamma = np.where(where_zero, 0, np.arctan2(mats[..., 1, 0], mats[..., 1, 1]))
        alpha = np.where(where_zero, 0, np.arctan2(mats[..., 0, 2], mats[..., 2, 2]))
        zero_terms = np.arctan2(-multiplier * mats[..., 2, 0], mats[..., 0, 0])
    elif convention == "yzx":
        beta = np.arcsin(mats[..., 1, 0])
        multiplier = mats[..., 1, 0] if axis_type == "extrinsic" else 1
        where_zero = np.isclose(np.cos(beta), 0, atol=atol)

        gamma = np.where(where_zero, 0, np.arctan2(-mats[..., 1, 2], mats[..., 1, 1]))
        alpha = np.where(where_zero, 0, np.arctan2(-mats[..., 2, 0], mats[..., 0, 0]))
        zero_terms = np.arctan2(multiplier * mats[..., 0, 2], mats[..., 2, 2])
    elif convention == "zyx":
        beta = np.arcsin(-mats[..., 2, 0])
        where_zero = np.isclose(np.cos(beta), 0, atol=atol)

        gamma = np.where(where_zero, 0, np.arctan2(mats[..., 2, 1], mats[..., 2, 2]))
        alpha = np.where(where_zero, 0, np.arctan2(mats[..., 1, 0], mats[..., 0, 0]))
        zero_terms = np.arctan2(-mats[..., 0, 1], mats[..., 1, 1])
    elif convention == "zxy":
        beta = np.arcsin(mats[..., 2, 1])
        multiplier = mats[..., 2, 1] if axis_type == "extrinsic" else 1
        where_zero = np.isclose(np.cos(beta), 0, atol=atol)

        gamma = np.where(where_zero, 0, np.arctan2(-mats[..., 2, 0], mats[..., 2, 2]))
        alpha = np.where(where_zero, 0, np.arctan2(-mats[..., 0, 1], mats[..., 1, 1]))
        zero_terms = np.arctan2(multiplier * mats[..., 1, 0], mats[..., 0, 0])
    else:
        raise ValueError("Unknown convention selected!")

    # For extrinsic, swap back alpha and gamma.
    if axis_type == "extrinsic":
        tmp = alpha
        alpha = gamma
        gamma = tmp

    # By convention, the zero terms that we calculate are always based on
    # setting gamma to zero and applying to alpha. We assign them after the
    # fact to enable the memcopy-free swap of alpha and gamma for extrinsic
    # angles. For Python 2 compatibility, we need to index appropriately.
    try:
        alpha[where_zero] = zero_terms[where_zero]
    except IndexError:
        # This is necessary for Python 2 compatibility and limitations with the
        # indexing behavior. Since the only possible case is a single set of
        # inputs, we can just skip any indexing and overwrite directly if
        # needed.
        if where_zero:
            alpha = zero_terms
    return np.stack((alpha, beta, gamma), axis=-1)


def from_matrix(mat, require_orthogonal=True):
    r"""Convert the rotation matrices mat to quaternions.

    This method uses the algorithm described by Bar-Itzhack in [Itzhack00]_.
    The idea is to construct a matrix K whose largest eigenvalue corresponds
    to the desired quaternion. One of the strengths of the algorithm is that
    for nonorthogonal matrices it gives the closest quaternion representation
    rather than failing outright.

    .. [Itzhack00] Itzhack Y. Bar-Itzhack.  "New Method for Extracting the
        Quaternion from a Rotation Matrix", Journal of Guidance, Control, and
        Dynamics, Vol. 23, No. 6 (2000), pp. 1085-1087
        https://doi.org/10.2514/2.4654

    Args:
        mat ((..., 3, 3) :class:`numpy.ndarray`):
            An array of rotation matrices.
        require_orthogonal (bool):
            Whether to require that the input matrices are orthogonal.

    Returns:
        (..., 4) :class:`numpy.ndarray`: The corresponding rotation quaternions.

    Example::

        >>> rowan.from_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        array([ 1., -0., -0., -0.])
    """
    mat = np.asarray(mat)
    if require_orthogonal and not np.allclose(np.linalg.det(mat), 1):
        raise ValueError(
            "Not all of your matrices are orthogonal. \
Please ensure that there are no improper rotations. \
If this was intentional, set require_orthogonal to \
False when calling this function.",
        )

    K = np.zeros(mat.shape[:-2] + (4, 4))
    K[..., 0, 0] = mat[..., 0, 0] - mat[..., 1, 1] - mat[..., 2, 2]
    K[..., 0, 1] = mat[..., 1, 0] + mat[..., 0, 1]
    K[..., 0, 2] = mat[..., 2, 0] + mat[..., 0, 2]
    K[..., 0, 3] = mat[..., 1, 2] - mat[..., 2, 1]
    K[..., 1, 0] = mat[..., 1, 0] + mat[..., 0, 1]
    K[..., 1, 1] = mat[..., 1, 1] - mat[..., 0, 0] - mat[..., 2, 2]
    K[..., 1, 2] = mat[..., 2, 1] + mat[..., 1, 2]
    K[..., 1, 3] = mat[..., 2, 0] - mat[..., 0, 2]
    K[..., 2, 0] = mat[..., 2, 0] + mat[..., 0, 2]
    K[..., 2, 1] = mat[..., 2, 1] + mat[..., 1, 2]
    K[..., 2, 2] = mat[..., 2, 2] - mat[..., 0, 0] - mat[..., 1, 1]
    K[..., 2, 3] = mat[..., 0, 1] - mat[..., 1, 0]
    K[..., 3, 0] = mat[..., 1, 2] - mat[..., 2, 1]
    K[..., 3, 1] = mat[..., 2, 0] - mat[..., 0, 2]
    K[..., 3, 2] = mat[..., 0, 1] - mat[..., 1, 0]
    K[..., 3, 3] = mat[..., 0, 0] + mat[..., 1, 1] + mat[..., 2, 2]
    K = K / 3.0

    _, v = np.linalg.eigh(K)
    # The conventions in the paper are very confusing for quaternions in terms
    # of the order of the components
    return np.concatenate((v[..., -1, -1, np.newaxis], -v[..., :-1, -1]), axis=-1)


def to_matrix(q, require_unit=True):
    r"""Convert quaternions into rotation matrices.

    Uses the conversion described on `Wikipedia
    <https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix>`_.

    Args:
        q ((..., 4) :class:`numpy.ndarray`):
            An array of quaternions.
        require_unit (bool):
            Whether to require that the input quaternions are unit quaternions.

    Returns:
        (..., 3, 3) :class:`numpy.ndarray`: The corresponding rotation matrices.

    Example::

        >>> rowan.to_matrix([1, 0, 0, 0])
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])
    """
    q = np.asarray(q)

    s = norm(q)
    if np.any(s == 0.0):
        raise ZeroDivisionError("At least one element of q has approximately zero norm")
    if require_unit and not np.allclose(s, 1.0):
        raise ValueError(
            "Not all quaternions in q are unit quaternions. \
If this was intentional, please set require_unit to False when \
calling this function.",
        )
    m = np.empty(q.shape[:-1] + (3, 3))
    s **= -1.0  # For consistency with Wikipedia notation
    m[..., 0, 0] = 1.0 - 2 * s * (q[..., 2] ** 2 + q[..., 3] ** 2)
    m[..., 0, 1] = 2 * (q[..., 1] * q[..., 2] - q[..., 3] * q[..., 0])
    m[..., 0, 2] = 2 * (q[..., 1] * q[..., 3] + q[..., 2] * q[..., 0])
    m[..., 1, 0] = 2 * (q[..., 1] * q[..., 2] + q[..., 3] * q[..., 0])
    m[..., 1, 1] = 1.0 - 2 * (q[..., 1] ** 2 + q[..., 3] ** 2)
    m[..., 1, 2] = 2 * (q[..., 2] * q[..., 3] - q[..., 1] * q[..., 0])
    m[..., 2, 0] = 2 * (q[..., 1] * q[..., 3] - q[..., 2] * q[..., 0])
    m[..., 2, 1] = 2 * (q[..., 2] * q[..., 3] + q[..., 1] * q[..., 0])
    m[..., 2, 2] = 1.0 - 2 * (q[..., 1] ** 2 + q[..., 2] ** 2)
    return m


def from_axis_angle(axes, angles):
    r"""Find quaternions to rotate a specified angle about a specified axis.

    All angles are assumed to be **counterclockwise** rotations about the axis.

    Args:
        axes ((..., 3) :class:`numpy.ndarray`):
            An array of vectors (the axes).
        angles (float or (..., 1) :class:`numpy.ndarray`):
            An array of angles in radians. Will be broadcasted to match shape of axes
            as needed.

    Returns:
        (..., 4) :class:`numpy.ndarray`: The corresponding rotation quaternions.

    Example::

        >>> import numpy as np
        >>> rowan.from_axis_angle([[1, 0, 0]], np.pi/3)
        array([[0.8660254, 0.5      , 0.       , 0.       ]])
    """
    axes = np.asarray(axes)

    # First reshape angles and compute the half angle
    bc = np.broadcast(angles, axes[..., 0])
    angles = np.broadcast_to(angles, bc.shape)[..., np.newaxis]
    axes = np.broadcast_to(axes, bc.shape + (3,))
    ha = angles / 2.0

    # Normalize the vector
    u = _normalize_vec(axes)

    # Compute the components of the quaternions
    scalar_comp = np.cos(ha)
    vec_comp = np.sin(ha) * u

    return np.concatenate((scalar_comp, vec_comp), axis=-1)


def to_axis_angle(q):
    r"""Convert the quaternions in q to axis-angle representations.

    The output angles are **counterclockwise** rotations about the axis.

    Args:
        q ((..., 4) :class:`numpy.ndarray`): An array of quaternions.

    Returns:
        tuple[(..., 3) :class:`numpy.ndarray`, (...) :class:`numpy.ndarray`]:
            The axes and the angles (in radians).

    Example::

        >>> rowan.to_axis_angle([[1, 0, 0, 0]])
        (array([[0., 0., 0.]]), array([0.]))
    """
    q = np.asarray(q)
    _validate_unit(q)

    angles = 2 * np.atleast_1d(np.arccos(q[..., 0]))
    sines = np.sin(angles / 2)
    # Avoid divide by zero issues; these values will not be used
    sines[sines == 0] = 1
    axes = np.where(
        angles[..., np.newaxis] != 0,
        q[..., 1:] / sines[..., np.newaxis],
        0,
    )

    return axes, angles


def equal(p, q):
    r"""Check whether two sets of quaternions are equal.

    This function is a simple wrapper that checks array
    equality and then aggregates along the quaternion axis.

    Args:
        p ((..., 4) :class:`numpy.ndarray`): First array of quaternions.
        q ((..., 4) :class:`numpy.ndarray`): Second array of quaternions.

    Returns:
        (...) :class:`numpy.ndarray` of bool: Whether ``p`` and ``q`` are equal.

    Example::

        >>> rowan.equal([1, 0, 0, 0], [1, 0, 0, 0])
        True
    """
    return np.all(p == q, axis=-1)


def not_equal(p, q):
    r"""Check whether two sets of quaternions are not equal.

    This function is a simple wrapper that checks array
    equality and then aggregates along the quaternion axis.

    Args:
        p ((..., 4) :class:`numpy.ndarray`): First array of quaternions.
        q ((..., 4) :class:`numpy.ndarray`): Second array of quaternions.

    Returns:
        (...) :class:`numpy.ndarray` of bool: Whether ``p`` and ``q`` are unequal.

    Example::

        >>> rowan.not_equal([-1, 0, 0, 0], [1, 0, 0, 0])
        True
    """
    return np.any(p != q, axis=-1)


def isnan(q):
    r"""Test element-wise for NaN quaternions.

    A quaternion is defined as NaN if any elements are NaN.

    Args:
        q ((..., 4) :class:`numpy.ndarray`): Array of quaternions.

    Returns:
        (...) :class:`numpy.ndarray` of bool: Whether ``q`` is NaN.

    Example::

        >>> import numpy as np
        >>> rowan.isnan([np.nan, 0, 0, 0])
        True
    """
    return np.any(np.isnan(q), axis=-1)


def isinf(q):
    r"""Test element-wise for infinite quaternions.

    A quaternion is defined as infinite if any elements are infinite.

    Args:
        q ((..., 4) :class:`numpy.ndarray`): Array of quaternions

    Returns:
        (...) :class:`numpy.ndarray` of bool: Whether ``q`` is infinite.

    Example::

        >>> import numpy as np
        >>> rowan.isinf([np.nan, 0, 0, 0])
        False
    """
    return np.any(np.isinf(q), axis=-1)


def isfinite(q):
    r"""Test element-wise for finite quaternions.

    A quaternion is defined as finite if all elements are finite.

    Args:
        q ((..., 4) :class:`numpy.ndarray`): Array of quaternions.

    Returns:
        (...) :class:`numpy.ndarray` of bool: Whether ``q`` is finite.

    Example::

        >>> rowan.isfinite([1, 0, 0, 0])
        True
    """
    return np.all(np.isfinite(q), axis=-1)


def allclose(p, q, **kwargs):  # noqa: D417
    r"""Check whether two sets of quaternions are all close.

    This is a direct wrapper of the corresponding NumPy function.

    Args:
        p ((..., 4) :class:`numpy.ndarray`): First array of quaternions.
        q ((..., 4) :class:`numpy.ndarray`): Second array of quaternions.
        \*\*kwargs: Keyword arguments to pass to np.allclose.

    Returns:
        bool: Whether all of ``p`` and ``q`` are close.

    Example::

        >>> rowan.allclose([1, 0, 0, 0], [1, 0, 0, 0])
        True
    """
    return np.allclose(p, q, **kwargs)


def isclose(p, q, **kwargs):  # noqa: D417
    r"""Element-wise check of whether two sets of quaternions are close.

    This function is a simple wrapper that checks using the
    corresponding NumPy function and then aggregates along
    the quaternion axis.

    Args:
        p ((..., 4) :class:`numpy.ndarray`): First array of quaternions.
        q ((..., 4) :class:`numpy.ndarray`): Second array of quaternions.
        \*\*kwargs: Keyword arguments to pass to np.isclose.

    Returns:
        (...) :class:`numpy.ndarray` of bool:
            Whether ``p`` and ``q`` are close element-wise.

    Example::

        >>> rowan.isclose([[1, 0, 0, 0]], [[1, 0, 0, 0]])
        array([ True])
    """
    return np.all(np.isclose(p, q, **kwargs), axis=-1)
