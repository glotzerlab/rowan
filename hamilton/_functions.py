# Copyright (c) 2018 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
"""Submodule containing all standard functions"""

import numpy as np
import warnings

def conjugate(q):
    R"""Returns the conjugate of quaternion array q

    Args:
        q ((...,4) np.array): First set of quaternions

    Returns:
        An (...,4) np.array containing the conjugates of q

    Example::

        q_star = conjugate(q)
    """
    # Don't use asarray to avoid modifying in place
    conjugate = np.array(q)
    conjugate[..., 1:] *= -1
    return conjugate


def multiply(qi, qj):
    R"""Multiplies the quaternions in the array qi by those in qj

    Args:
        qi ((...,4) np.array): First set of quaternions
        qj ((...,4) np.array): Second set of quaternions

    Returns:
        An (...,4) np.array containing the products
        of row i of qi with column j of qj

    Example::

        qi = np.array([[1, 0, 0, 0]])
        qj = np.array([[1, 0, 0, 0]])
        prod = multiply(qi, qj)
    """
    qi = np.asarray(qi)
    qj = np.asarray(qj)
    if not qi.shape == qj.shape:
        raise ValueError("The two arrays must be the same size!")

    output = np.empty(qi.shape)

    output[..., 0] = qi[..., 0] * qj[..., 0] - \
        np.sum(qi[..., 1:] * qj[..., 1:], axis=-1)
    output[..., 1:] = (qi[..., 0, np.newaxis] * qj[..., 1:] +
                       qj[..., 0, np.newaxis] * qi[..., 1:] +
                       np.cross(qi[..., 1:], qj[..., 1:]))
    return output


def norm(q):
    R"""Trivial reimplementation of norm for both quaternions and vectors

    Args:
        q ((...,4) np.array): Quaternions to normalize

    Returns:
        A (...) np.array containing the norms for qi in q

    Example::

        q = np.random.rand(10, 4)
        norms = norm(q)
    """
    q = np.asarray(q)
    return np.linalg.norm(q, axis=-1)


def normalize(q):
    R"""Normalize quaternion or vector input

    Args:
        q ((...,{3,4}) np.array): Array of quaternions/vectors to normalize

    Returns:
        An (...,{3,4}) np.array containing the unit quaternions qi/norm(qi)

    Example::

        q = np.random.rand(10, 4)
        u = normalize(q)
    """
    q = np.asarray(q)
    norms = norm(q)
    return q / norms[..., np.newaxis]


def rotate(q, v):
    R"""Performs an element-wise rotation of the vectors
    v by the quaternions q.
    The shapes of the two arrays must conform up to the
    last dimension.

    Args:
        q ((...,4) np.array): First set of quaternions
        v ((...,3) np.array): First set of quaternions

    Returns:
        An (...,3) np.array of the vectors in v rotated by q

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


def _vector_bisector(v1, v2):
    R"""Find the vector bisecting v1 and v2

    Args:
        v1 ((...,3) np.array): First vector
        v2 ((...,3) np.array): Second vector

    Returns:
        The vector that bisects the angle between v1 and v2

    """

    # Check that the vectors are reasonable
    if len(v1.shape) == 1:
        v1 = v1[np.newaxis, :]
    if len(v2.shape) == 2:
        v2 = v2[np.newaxis, :]
    return normalize(normalize(v1) + normalize(v2))


def about_axis(v, theta):
    R"""Find the quaternions corresponding to rotations about
    the axes v by angles theta

    Args:
        v ((...,3) np.array): Axes to rotate about
        theta (float or (...) np.array): Angle (in radians).
            Will be broadcast to match shape of v as needed

    Returns:
        An (...,4) np.array of the requested rotation quaternions

    Example::

        import numpy as np
        axis = np.array([[1, 0, 0]])
        ang = np.pi/3
        quat = about_axis(axis, ang)
    """
    v = np.asarray(v)

    # First reshape theta and compute the half angle
    theta = np.broadcast_to(theta, v.shape[:-1])[..., np.newaxis]
    ha = theta / 2.0

    # Normalize the vector
    u = normalize(v)

    # Compute the components of the quaternions
    scalar_comp = np.cos(ha)
    vec_comp = np.sin(ha) * u

    return np.concatenate((scalar_comp, vec_comp), axis=-1)


def vector_vector_rotation(v1, v2):
    R"""Find the quaternion to rotate v1 onto v2

    Args:
        v1 ((...,3) np.array): Vector to rotate
        v2 ((...,3) np.array): Desired vector

    """
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    return about_axis(_vector_bisector(v1, v2), np.pi)


def from_euler(angles, convention = 'zyx', axis_type = 'intrinsic'):
    R"""Convert Euler angles to quaternion

    Args:
        angles ((...,3) np.array): Array whose last dimension
            (of size 3) is (alpha, beta, gamma)
        convention (str): One of the 6 valid conventions zxz,
            xyx, yzy, zyz, xzx, yxy
        axes (str): Whether to use extrinsic or intrinsic

    Returns:
        An (..., 4) np.array containing the converted quaternions

    For generality, the rotations are computed by composing a sequence
    of quaternions corresponding to axis-angle rotations. While more
    efficient implementations are possible, this method is more flexible
    for getting all types.

    Example::

        rands = np.random.rand(100, 3)
        alpha, beta, gamma = rands.T
        ql.from_euler(alpha, beta, gamma)
    """
    angles = np.asarray(angles)
    convention = convention.lower()
    #TODO: USE THE CODE HERE AS A WAY TO DETERMINE WHERE BROADCASTING CAN BE MADE MORE EFFICIENT THROUGHOUT THE MODULE

    basis_axes = {
            'x': np.array([1, 0, 0]),
            'y': np.array([0, 1, 0]),
            'z': np.array([0, 0, 1]),
            }
    # Temporary method to ensure shapes conform
    for ax, vec in basis_axes.items():
        basis_axes[ax] = np.broadcast_to(
                            vec,
                            (*angles.shape[:-1],
                                vec.shape[-1])
                            )

    # Split by convention, the easiest
    rotations = []
    if axis_type == 'extrinsic':
        # Loop over the axes and add each rotation
        for i, char in enumerate(convention):
            ax = basis_axes[char]
            rotations.append(from_axis_angle(ax, angles[..., i]))
    elif axis_type == 'intrinsic':
        for i, char in enumerate(convention):
            ax = basis_axes[char]
            rotations.append(from_axis_angle(ax, angles[..., i]))
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


def from_euler_old(angles):
    R"""Convert Euler angles to quaternion (3-2-1 convention)

    Args:
        angles ((...,3) np.array): Array whose last dimension
            (of size 3) is (alpha, beta, gamma)

    Returns:
        An (..., 4) np.array containing the converted quaternions

    Standard numpy broadcasting is used to compute the quaternions
    along the last dimension of the angle arrays.

    Note:
        Derived from injavis implementation

    Example::

        rands = np.random.rand(100, 3)
        alpha, beta, gamma = rands.T
        ql.from_euler(alpha, beta, gamma)
    """
    angles = np.asarray(angles)

    try:
        angles *= 0.5
    except TypeError:
        # Can't multiply integral types in place, but avoid copying if possible
        angles = angles * 0.5

    c = np.cos(angles)
    s = np.sin(angles)

    r = np.prod(c, axis=-1) + np.prod(s, axis=-1)
    i = s[..., 0] * c[..., 1] * c[..., 2] - c[..., 0] * s[..., 1] * s[..., 2]
    j = c[..., 0] * s[..., 1] * c[..., 2] + s[..., 0] * c[..., 1] * s[..., 2]
    k = c[..., 0] * c[..., 1] * s[..., 2] - s[..., 0] * s[..., 1] * c[..., 2]

    return np.stack([r, i, j, k], axis=-1)


def to_euler(q, convention = 'zyx', axis_type = 'intrinsic'):
    R"""Convert quaternions to Euler angles

    Euler angles are returned in the sequence provided, so in the
    default case ('zyx') the angles are for a rotation
    :math:`Z(\alpha) Y(\beta) X(\gamma)`.

    For simplicity, quaternions are converted to matrices, which are
    then converted to their Euler angle representations. All equations
    for intrinsic rotations are derived by considering compositions
    of three elemental rotations about the three Cartesian axes:

    .. math::
        :nowrap:

        begin{eqnarray*}
        R_x(\theta)  =& \left(\begin{array}{ccc}
                            1   & 0             & 0 \\
                            0   & cos \theta    & -sin \theta \\
                            0   & sin \theta    & cos \theta    \\
                         \end{array}\right)
        R_y(\theta)  =& \left(\begin{array}{ccc}
                            cos \theta   & 0        & sin \theta \\
                            0            & 1        &  0\\
                            -sin \theta  & 1        & cos \theta    \\
                         \end{array}\right)
        R_z(\theta)  =& \left(\begin{array}{ccc}
                            cos \theta  & -sin \theta   & 0 \\
                            sin \theta  & cos \theta    & 0 \\
                            0           & 0             & 1 \\
                         \end{array}\right)
        \end{eqnarray*}

    For intrinsic rotations, the order of rotations matches the order
    of matrices *i.e.* the z-y'-x'' convention (yaw, pitch, roll)
    corresponds to the multiplication of matrices ZYX. For more
    information, see the Wikipedia page for Euler angles (specifically
    the section on converting between representations):
    https://en.wikipedia.org/wiki/Euler_angles.

    Extrinsic rotations are then derived by considering the
    composition of intrinsic rotations in the opposite order.
    For proof of the relationship between intrinsic and extrinsic
    rotations, see the Wikipedia page on Davenport chained rotations:
    https://en.wikipedia.org/wiki/Davenport_chained_rotations

    It may be more natural to think of extrinsic rotations as
    applying matrix rotations in the proper order, *i.e.* for standard
    right-handed coordinate systems and the application of rotations
    viewed as pre-multiplication of column vectors, an extrinic
    rotation about x-y-z is the multiplication of the rotation matrices
    ZYX since X is applied first, then Y, then Z. This order is
    reversed for intrinsic rotations, so the matrix order is identical
    to the order in the name (x-y'-z'' rotation represents XYZ).

    Args:
        q ((...,4) np.array): Quaternions to transform
        convention (str): One of the 6 valid conventions zxz,
            xyx, yzy, zyz, xzx, yxy
        axes (str): Whether to use extrinsic or intrinsic

    Returns:
        A (..., 3) np.array with Euler angles (alpha, beta, gamma)
        as the last dimension (in radians)

    Example::

        rands = np.random.rand(100, 3)
        alpha, beta, gamma = rands.T
        ql.from_euler(alpha, beta, gamma)
        alpha_return, beta_return, gamma_return = ql.to_euler(full)

    """
    q = np.asarray(q)

    mats = to_matrix(q)

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

    return np.stack((alpha, beta, gamma), axis = -1)


def to_euler_old(q):
    R"""Convert quaternions to Euler angles (3-2-1 convention)

    Args:
        q ((...,4) np.array): Quaternions to transform

    Returns:
        A (..., 3) np.array with Euler angles (alpha, beta, gamma)
        as the last dimension (in radians)

    Note:
        Derived from injavis implementation

    Example::

        rands = np.random.rand(100, 3)
        alpha, beta, gamma = rands.T
        ql.from_euler(alpha, beta, gamma)
        alpha_return, beta_return, gamma_return = ql.to_euler(full)

    """
    q = np.asarray(q)

    r = q[..., 0, np.newaxis]
    i = q[..., 1, np.newaxis]
    j = q[..., 2, np.newaxis]
    k = q[..., 3, np.newaxis]

    q00 = r * r
    q11 = i * i
    q22 = j * j
    q33 = k * k
    q01 = r * i
    q02 = r * j
    q03 = r * k
    q12 = i * j
    q13 = i * k
    q23 = j * k

    alpha = np.arctan2(2.0 * (q01 + q23), q00 - q11 - q22 + q33)
    beta = np.arcsin(2.0 * (q02 - q13))
    gamma = np.arctan2(2.0 * (q03 + q12), q00 + q11 - q22 - q33)

    alpha[np.isnan(alpha)] = np.pi / 2
    beta[np.isnan(beta)] = np.pi / 2
    gamma[np.isnan(gamma)] = np.pi / 2

    return np.concatenate((alpha, beta, gamma), axis=-1)


def from_matrix(mat, require_orthogonal=True):
    R"""Convert the rotation matrices mat to quaternions

    Uses the algorithm described in this paper by Bar-Itzhack
    <https://doi.org/10.2514/2.4654>. The idea is to construct a
    matrix K whose largest eigenvalue corresponds to the desired
    quaternion. One of the strengths of the algorithm is that for
    nonorthogonal matrices it gives the closest quaternion
    representation rather than failing outright.

    Args:
        mat ((...,3,3) np.array): An array of rotation matrices

    Returns:
        An (..., 4) np.array containing the quaternion representations
        of the elements of mat (i.e. the same elements of SO(3))
    """
    mat = np.asarray(mat)
    if not np.allclose(np.linalg.det(mat), 1) and require_orthogonal:
        warnings.warn(
            "Not all of your matrices are orthogonal. \
Please ensure that there are no improper rotations. \
If this was intentional, please set require_orthogonal \
to False when calling this function.",
            UserWarning)

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
    R"""Convert the quaternions in q to rotation matrices.

    Uses the conversion described on Wikipedia
    <https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix>

    Args:
        q ((...,4) np.array): An array of quaternions

    Returns:
        The (..., 3, 3) np.array containing the matrix representations
        of the elements of q (i.e. the same elements of SO(3))
    """
    q = np.asarray(q)

    s = norm(q)
    if np.any(s == 0.0):
        raise ZeroDivisionError(
            "At least one element of q has approximately zero norm")
    else:
        if not np.allclose(s, 1.0):
            if require_unit:
                raise RuntimeError(
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
    R"""Generate quaternions from axes and angles element-wise

    Args:
        axes ((...,3) np.array): An array of vectors (the axes)
        angles ((...,1) np.array): An array of angles in radians.
            If the last dimension is not singular one will be appended
            to conform to the axes array.

    Returns:
        An (..., 4) np.array containing the quaternions equivalent
        to rotating angles about axes
    """
    axes = np.asarray(axes)

    # Ensure appropriate shape for angles array
    angles = np.atleast_1d(np.asarray(angles))
    if not angles.shape[-1] == 1:
        angles = angles[..., np.newaxis]

    if axes.shape[:-1] != angles.shape[:-1]:
        raise ValueError("The input arrays must conform in dimension")

    # Ensure conforming shapes
    if not angles.shape[-1] == 1:
        angles = angles[..., np.newaxis]
    return np.concatenate(
            (np.cos(angles/2), axes*np.sin(angles/2)),
            axis = -1)


def to_axis_angle(q):
    R"""Convert the quaternions in q to axis angle representations

    Args:
        q ((...,4) np.array): An array of quaternions

    Returns:
        A tuple of np.arrays (axes, angles) where axes has
        shape (..., 3) and angles has shape (..., 1). The
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
