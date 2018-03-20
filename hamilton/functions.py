# Copyright (c) 2018 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
"""Submodule containing all standard functions"""
from __future__ import division, print_function, absolute_import

import numpy as np


def conjugate(q):
    R"""Conjugates an array of quaternions

    Args:
        q ((...,4) np.array): First set of quaternions

    Returns:
        An array containing the conjugates of q

    Example::

        q_star = conjugate(q)
    """
    # Don't use asarray to avoid modifying in place
    conjugate = np.array(q)
    conjugate[..., 1:] *= -1
    return conjugate


def multiply(qi, qj):
    R"""Multiplies two arrays of quaternions

    Note that quaternion multiplication is generally non-commutative.

    Args:
        qi ((...,4) np.array): First set of quaternions
        qj ((...,4) np.array): Second set of quaternions

    Returns:
        An array containing the products of row i of qi
        with column j of qj

    Example::

        qi = np.array([[1, 0, 0, 0]])
        qj = np.array([[1, 0, 0, 0]])
        prod = multiply(qi, qj)
    """
    qi = np.asarray(qi)
    qj = np.asarray(qj)

    output = np.empty(np.broadcast(qi, qj).shape)

    output[..., 0] = qi[..., 0] * qj[..., 0] - \
        np.sum(qi[..., 1:] * qj[..., 1:], axis=-1)
    output[..., 1:] = (qi[..., 0, np.newaxis] * qj[..., 1:] +
                       qj[..., 0, np.newaxis] * qi[..., 1:] +
                       np.cross(qi[..., 1:], qj[..., 1:]))
    return output


def norm(q):
    R"""Compute the quaternion norm

    Args:
        q ((...,4) np.array): Quaternions to find norms for

    Returns:
        An array containing the norms for qi in q

    Example::

        q = np.random.rand(10, 4)
        norms = norm(q)
    """
    q = np.asarray(q)
    return np.linalg.norm(q, axis=-1)


def normalize(q):
    R"""Normalize quaternions

    Args:
        q ((...,4) np.array): Array of quaternions to normalize

    Returns:
        An array containing the unit quaternions q/norm(q)

    Example::

        q = np.random.rand(10, 4)
        u = normalize(q)
    """
    q = np.asarray(q)
    norms = norm(q)
    return q / norms[..., np.newaxis]


def rotate(q, v):
    R"""Rotate a list of vectors by a corresponding set of quaternions

    The shapes of the two arrays must conform up to the last dimension.

    Args:
        q ((...,4) np.array): First set of quaternions
        v ((...,3) np.array): First set of quaternions

    Returns:
        An array of the vectors in v rotated by q

    Example::

        q = np.random.rand(1, 4)
        v = np.random.rand(1, 3)
        v_rot = rotate(q, v)
    """
    q = np.asarray(q)
    v = np.asarray(v)

    # Convert vector to quaternion representation
    quat_v = np.concatenate((np.zeros(v.shape[:-1] + (1,)), v), axis=-1)
    return multiply(q, multiply(quat_v, conjugate(q)))[..., 1:]


def _normalize_vec(v):
    """Helper function to normalize vectors"""
    return v/np.linalg.norm(v, axis=-1)[..., np.newaxis]


def _vector_bisector(v1, v2):
    R"""Find the vector bisecting two vectors

    Args:
        v1 ((...,3) np.array): First vector
        v2 ((...,3) np.array): Second vector

    Returns:
        The vector that bisects the angle between v1 and v2
    """

    return _normalize_vec(_normalize_vec(v1) + _normalize_vec(v2))


def vector_vector_rotation(v1, v2):
    R"""Find the quaternion to rotate one vector onto another

    Args:
        v1 ((...,3) np.array): Vector to rotate
        v2 ((...,3) np.array): Desired vector

    Returns:
        Array (..., 4) of quaternions that rotate v1 onto v2.
    """
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    return from_axis_angle(_vector_bisector(v1, v2), np.pi)


def from_euler(alpha, beta, gamma, convention='zyx',
               axis_type='intrinsic'):
    R"""Convert Euler angles to quaternions

    For generality, the rotations are computed by composing a sequence of
    quaternions corresponding to axis-angle rotations. While more efficient
    implementations are possible, this method was chosen to prioritize
    flexibility since it works for essentially arbitrary Euler angles as
    long as intrinsic and extrinsic rotations are not intermixed.

    Args:
        alpha ((...) np.array): Array of :math:`\alpha` values
        beta ((...) np.array): Array of :math:`\beta` values
        gamma ((...) np.array): Array of :math:`\gamma` values
        convention (str): One of the 12 valid conventions xzx, xyx,
            yxy, yzy, zyz, zxz, xzy, xyz, yxz, yzx, zyx, zxy
        axes (str): Whether to use extrinsic or intrinsic rotations

    Returns:
        An array containing the converted quaternions

    Example::

        rands = np.random.rand(100, 3)
        alpha, beta, gamma = rands.T
        ql.from_euler(alpha, beta, gamma)
    """
    angles = np.broadcast_arrays(alpha, beta, gamma)

    convention = convention.lower()

    if len(convention) > 3 or (set(convention) - set('xyz')):
        raise ValueError("All acceptable conventions must be 3 \
character strings composed only of x, y, and z")

    basis_axes = {
        'x': np.array([1, 0, 0]),
        'y': np.array([0, 1, 0]),
        'z': np.array([0, 0, 1]),
    }
    # Temporary method to ensure shapes conform
    for ax, vec in basis_axes.items():
        basis_axes[ax] = np.broadcast_to(
            vec,
            angles[0].shape + (vec.shape[-1],)
        )

    # Split by convention, the easiest
    rotations = []
    if axis_type == 'extrinsic':
        # Loop over the axes and add each rotation
        for i, char in enumerate(convention):
            ax = basis_axes[char]
            rotations.append(from_axis_angle(ax, angles[i]))
    elif axis_type == 'intrinsic':
        for i, char in enumerate(convention):
            ax = basis_axes[char]
            rotations.append(from_axis_angle(ax, angles[i]))
            # Rotate the bases as well
            for key, value in basis_axes.items():
                basis_axes[key] = rotate(
                    rotations[-1],
                    value
                )
    else:
        raise ValueError("Only valid axis_types are intrinsic and extrinsic")

    # Compose the total rotation
    final_rotation = np.broadcast_to(
        np.array([1, 0, 0, 0]),
        rotations[0].shape
    )
    for q in rotations:
        final_rotation = multiply(q, final_rotation)

    return final_rotation


def to_euler(q, convention='zyx', axis_type='intrinsic'):
    R"""Convert quaternions to Euler angles

    Euler angles are returned in the sequence provided, so in, *e.g.*,
    the default case ('zyx'), the angles returned are for a rotation
    :math:`Z(\alpha) Y(\beta) X(\gamma)`.

    .. note::

        In all cases, the :math:`\alpha` and :math:`\gamma` angles are
        between :math:`\pm \pi`. For proper Euler angles, :math:`\beta`
        is between :math:`0` and :math:`pi` degrees. For Tait-Bryan
        angles, :math:`\beta` lies between :math:`\pm\pi/2`.

    For simplicity, quaternions are converted to matrices, which are
    then converted to their Euler angle representations. All equations
    for rotations are derived by considering compositions of the three
    elemental rotations about the three Cartesian axes:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        R_x(\theta)  =& \left(\begin{array}{ccc}
                            1   & 0             & 0 \\
                            0   & \cos \theta    & -\sin \theta \\
                            0   & \sin \theta    & \cos \theta    \\
                         \end{array}\right)\\
        R_y(\theta)  =& \left(\begin{array}{ccc}
                            \cos \theta   & 0        & \sin \theta \\
                            0            & 1        &  0\\
                            -\sin \theta  & 1        & \cos \theta    \\
                         \end{array}\right)\\
        R_z(\theta)  =& \left(\begin{array}{ccc}
                            \cos \theta  & -\sin \theta   & 0 \\
                            \sin \theta  & \cos \theta    & 0 \\
                            0           & 0             & 1 \\
                         \end{array}\right)\\
        \end{eqnarray*}

    Extrinsic rotations are represented by matrix multiplications in
    the proper order, so :math:`z-y-x` is represented by the
    multiplication :math:`XYZ` so that the system is rotated first
    about :math:`Z`, then about :math:`y`, then finally :math:`X`.
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

    Args:
        q ((...,4) np.array): Quaternions to transform
        convention (str): One of the 6 valid conventions zxz,
            xyx, yzy, zyz, xzx, yxy
        axes (str): Whether to use extrinsic or intrinsic

    Returns:
        An array with Euler angles :math:`(\alpha, \beta, \gamma)`
        as the last dimension (in radians)

    Example::

        rands = np.random.rand(100, 3)
        alpha, beta, gamma = rands.T
        ql.from_euler(alpha, beta, gamma)
        alpha_return, beta_return, gamma_return = ql.to_euler(full)

    """
    q = np.asarray(q)

    try:
        mats = to_matrix(q)
    except ValueError:
        raise ValueError(
            "Not all quaternions in q are unit quaternions.")
    except: # noqa E722
        raise

    if axis_type == 'intrinsic':
        # Have to hardcode the different possibilites.
        # Classical Euler angles
        if convention == 'xzx':
            alpha = np.arctan2(mats[..., 2, 0], mats[..., 1, 0])
            beta = np.arccos(mats[..., 0, 0])
            gamma = np.arctan2(mats[..., 0, 2], -mats[..., 0, 1])
        elif convention == 'xyx':
            alpha = np.arctan2(mats[..., 1, 0], -mats[..., 2, 0])
            beta = np.arccos(mats[..., 0, 0])
            gamma = np.arctan2(mats[..., 0, 1], mats[..., 0, 2])
        elif convention == 'yxy':
            alpha = np.arctan2(mats[..., 0, 1], mats[..., 2, 1])
            beta = np.arccos(mats[..., 1, 1])
            gamma = np.arctan2(mats[..., 1, 0], -mats[..., 1, 2])
        elif convention == 'yzy':
            alpha = np.arctan2(mats[..., 2, 1], -mats[..., 0, 1])
            beta = np.arccos(mats[..., 1, 1])
            gamma = np.arctan2(mats[..., 1, 2], mats[..., 1, 0])
        elif convention == 'zyz':
            alpha = np.arctan2(mats[..., 1, 2], mats[..., 0, 2])
            beta = np.arccos(mats[..., 2, 2])
            gamma = np.arctan2(mats[..., 2, 1], -mats[..., 2, 0])
        elif convention == 'zxz':
            alpha = np.arctan2(mats[..., 0, 2], -mats[..., 1, 2])
            beta = np.arccos(mats[..., 2, 2])
            gamma = np.arctan2(mats[..., 2, 0], mats[..., 2, 1])
        # Tait-Bryan angles
        elif convention == 'xzy':
            alpha = np.arctan2(mats[..., 2, 1], mats[..., 1, 1])
            beta = np.arcsin(-mats[..., 0, 1])
            gamma = np.arctan2(mats[..., 0, 2], mats[..., 0, 0])
        elif convention == 'xyz':
            alpha = np.arctan2(-mats[..., 1, 2], mats[..., 2, 2])
            beta = np.arcsin(mats[..., 0, 2])
            gamma = np.arctan2(-mats[..., 0, 1], mats[..., 0, 0])
        elif convention == 'yxz':
            alpha = np.arctan2(mats[..., 0, 2], mats[..., 2, 2])
            beta = np.arcsin(-mats[..., 1, 2])
            gamma = np.arctan2(mats[..., 1, 0], mats[..., 1, 1])
        elif convention == 'yzx':
            alpha = np.arctan2(-mats[..., 2, 0], mats[..., 0, 0])
            beta = np.arcsin(mats[..., 1, 0])
            gamma = np.arctan2(-mats[..., 1, 2], mats[..., 1, 1])
        elif convention == 'zyx':
            alpha = np.arctan2(mats[..., 1, 0], mats[..., 0, 0])
            beta = np.arcsin(-mats[..., 2, 0])
            gamma = np.arctan2(mats[..., 2, 1], mats[..., 2, 2])
        elif convention == 'zxy':
            alpha = np.arctan2(-mats[..., 0, 1], mats[..., 1, 1])
            beta = np.arcsin(mats[..., 2, 1])
            gamma = np.arctan2(-mats[..., 2, 0], mats[..., 2, 2])
        else:
            raise ValueError("Unknown convention selected!")
    elif axis_type == 'extrinsic':
        # For these, the matrix must be constructed in reverse order
        # e.g. Z(\alpha)Y'(\beta)Z''(\gamma) (where primes denote the
        # rotated frames) becomes the extrinsic rotation
        # Z(\gamma)Y(\beta)Z(\alpha).

        # Classical Euler angles
        if convention == 'xzx':
            alpha = np.arctan2(mats[..., 0, 2], -mats[..., 0, 1])
            beta = np.arccos(mats[..., 0, 0])
            gamma = np.arctan2(mats[..., 2, 0], mats[..., 1, 0])
        elif convention == 'xyx':
            alpha = np.arctan2(mats[..., 0, 1], mats[..., 0, 2])
            beta = np.arccos(mats[..., 0, 0])
            gamma = np.arctan2(mats[..., 1, 0], -mats[..., 2, 0])
        elif convention == 'yxy':
            alpha = np.arctan2(mats[..., 1, 0], -mats[..., 1, 2])
            beta = np.arccos(mats[..., 1, 1])
            gamma = np.arctan2(mats[..., 0, 1], mats[..., 2, 1])
        elif convention == 'yzy':
            alpha = np.arctan2(mats[..., 1, 2], mats[..., 1, 0])
            beta = np.arccos(mats[..., 1, 1])
            gamma = np.arctan2(mats[..., 2, 1], -mats[..., 0, 1])
        elif convention == 'zyz':
            alpha = np.arctan2(mats[..., 2, 1], -mats[..., 2, 0])
            beta = np.arccos(mats[..., 2, 2])
            gamma = np.arctan2(mats[..., 1, 2], mats[..., 0, 2])
        elif convention == 'zxz':
            alpha = np.arctan2(mats[..., 2, 0], mats[..., 2, 1])
            beta = np.arccos(mats[..., 2, 2])
            gamma = np.arctan2(mats[..., 0, 2], -mats[..., 1, 2])
        # Tait-Bryan angles
        elif convention == 'xzy':
            alpha = np.arctan2(-mats[..., 1, 2], mats[..., 1, 1])
            beta = np.arcsin(mats[..., 1, 0])
            gamma = np.arctan2(-mats[..., 2, 0], mats[..., 0, 0])
        elif convention == 'xyz':
            alpha = np.arctan2(mats[..., 2, 1], mats[..., 2, 2])
            beta = np.arcsin(-mats[..., 2, 0])
            gamma = np.arctan2(mats[..., 1, 0], mats[..., 0, 0])
        elif convention == 'yxz':
            alpha = np.arctan2(-mats[..., 2, 0], mats[..., 2, 2])
            beta = np.arcsin(mats[..., 2, 1])
            gamma = np.arctan2(-mats[..., 0, 1], mats[..., 1, 1])
        elif convention == 'yzx':
            alpha = np.arctan2(mats[..., 0, 2], mats[..., 0, 0])
            beta = np.arcsin(-mats[..., 0, 1])
            gamma = np.arctan2(mats[..., 2, 1], mats[..., 1, 1])
        elif convention == 'zyx':
            alpha = np.arctan2(-mats[..., 0, 1], mats[..., 0, 0])
            beta = np.arcsin(mats[..., 0, 2])
            gamma = np.arctan2(-mats[..., 1, 2], mats[..., 2, 2])
        elif convention == 'zxy':
            alpha = np.arctan2(mats[..., 1, 0], mats[..., 1, 1])
            beta = np.arcsin(-mats[..., 1, 2])
            gamma = np.arctan2(mats[..., 0, 2], mats[..., 2, 2])
        else:
            raise ValueError("Unknown convention selected!")
    else:
        raise ValueError("The axis type must be either extrinsic or intrinsic")

    return np.stack((alpha, beta, gamma), axis=-1)


def from_matrix(mat, require_orthogonal=True):
    R"""Convert the rotation matrices mat to quaternions

    Uses the algorithm described Bar-Itzhack described in this `paper
    <https://doi.org/10.2514/2.4654>`_. The idea is to construct a
    matrix K whose largest eigenvalue corresponds to the desired
    quaternion. One of the strengths of the algorithm is that for
    nonorthogonal matrices it gives the closest quaternion
    representation rather than failing outright.

    Args:
        mat ((...,3,3) np.array): An array of rotation matrices

    Returns:
        An array containing the quaternion representations
        of the elements of mat (i.e. the same elements of SO(3))
    """
    mat = np.asarray(mat)
    if require_orthogonal and not np.allclose(np.linalg.det(mat), 1):
        raise ValueError(
            "Not all of your matrices are orthogonal. \
Please ensure that there are no improper rotations. \
If this was intentional, set require_orthogonal to \
False when calling this function.")

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

    w, v = np.linalg.eigh(K)
    # The conventions in the paper are very confusing for quaternions in terms
    # of the order of the components
    return np.concatenate(
        (v[..., -1, -1, np.newaxis], -v[..., :-1, -1]), axis=-1)


def to_matrix(q, require_unit=True):
    R"""Convert quaternions into rotation matrices.

    Uses the conversion described on `Wikipedia
    <https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix>`_.

    Args:
        q ((...,4) np.array): An array of quaternions

    Returns:
        The array containing the matrix representations
        of the elements of q (i.e. the same elements of SO(3))
    """
    q = np.asarray(q)

    s = norm(q)
    if np.any(s == 0.0):
        raise ZeroDivisionError(
            "At least one element of q has approximately zero norm")
    elif require_unit and not np.allclose(s, 1.0):
        raise ValueError(
            "Not all quaternions in q are unit quaternions. \
If this was intentional, please set require_unit to False when \
calling this function.")
    m = np.empty(q.shape[:-1] + (3, 3))
    s **= -1.0  # For consistency with Wikipedia notation
    m[..., 0, 0] = 1.0 - 2 * s * (q[..., 2]**2 + q[..., 3]**2)
    m[..., 0, 1] = 2 * (q[..., 1] * q[..., 2] - q[..., 3] * q[..., 0])
    m[..., 0, 2] = 2 * (q[..., 1] * q[..., 3] + q[..., 2] * q[..., 0])
    m[..., 1, 0] = 2 * (q[..., 1] * q[..., 2] + q[..., 3] * q[..., 0])
    m[..., 1, 1] = 1.0 - 2 * (q[..., 1]**2 + q[..., 3]**2)
    m[..., 1, 2] = 2 * (q[..., 2] * q[..., 3] - q[..., 1] * q[..., 0])
    m[..., 2, 0] = 2 * (q[..., 1] * q[..., 3] - q[..., 2] * q[..., 0])
    m[..., 2, 1] = 2 * (q[..., 2] * q[..., 3] + q[..., 1] * q[..., 0])
    m[..., 2, 2] = 1.0 - 2 * (q[..., 1]**2 + q[..., 2]**2)
    return m


def from_axis_angle(axes, angles):
    R"""Find quaternions to rotate a specified angle about a specified axis

    Args:
        axes ((...,3) np.array): An array of vectors (the axes)
        angles (float or (...,1) np.array): An array of angles in radians.
            Will be broadcast to match shape of v as needed

    Returns:
        An array of the desired rotation quaternions

    Example::

        import numpy as np
        axis = np.array([[1, 0, 0]])
        ang = np.pi/3
        quat = about_axis(axis, ang)
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
    R"""Convert the quaternions in q to axis angle representations

    Args:
        q ((...,4) np.array): An array of quaternions

    Returns:
        A tuple of np.arrays (axes, angles) where axes has
        shape (...,3) and angles has shape (...,1). The
        angles are in radians
    """
    q = np.asarray(q)

    angles = 2*np.atleast_1d(np.arccos(q[..., 0]))
    sines = np.sin(angles/2)
    # Avoid divide by zero issues; these values will not be used
    sines[sines == 0] = 1
    axes = np.where(angles != 0,
                    q[..., 1:]/sines,
                    0)

    return axes, angles
