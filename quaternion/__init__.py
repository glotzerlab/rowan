
"""Simple quaternion library containing standard methods"""

import numpy as np
import warnings

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

__version__ = 0.1

def conjugate(q):
    R"""Returns the conjugate of quaternion array q

    Args:
        q ((...,4) np.array): First set of quaternions

    Returns:
        An (...,4) np.array containing the conjugates of q

    Example::

        q_star = conjugate(q)
    """
    conjugate = np.asarray(q)
    conjugate[..., 1:] *= -1
    return conjugate

def multiply(qi, qj):
    R"""Multiplies the quaternions in the array qi by those in qj

    Args:
        qi ((...,4) np.array): First set of quaternions
        qj ((...,4) np.array): Second set of quaternions

    Returns:
        An (...,4) np.array containing the products of row i of qi with column j of qj

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

    output[..., 0] = qi[..., 0] * qj[..., 0] - np.sum(qi[..., 1:] * qj[..., 1:], axis=-1)
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
    return np.linalg.norm(q, axis = -1)

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
    return q/norms[..., np.newaxis]

def rotate(q, v):
    R"""Rotates the vectors v by the quaternions q

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
    quat_v = np.concatenate((np.zeros(v.shape[:-1]+(1,)), v), axis = -1)
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
    #if not v1.shape == v2.shape == [1, 3]:
        #raise ValueError("The two inputs must both be 1x3 vectors")
    return normalize(normalize(v1) + normalize(v2))

def about_axis(v, theta):
    R"""Find the quaternions corresponding to rotations about the axes v by angles theta

    Args:
        v ((...,3) np.array): Axes to rotate about
        theta (float or (...) np.array): Angle (in radians). Will be broadcast to match shape of v as needed

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

    return np.concatenate((scalar_comp, vec_comp), axis = -1)

def vector_vector_rotation(v1, v2):
    R"""Find the quaternion to rotate v1 onto v2

    Args:
        v1 ((...,3) np.array): Vector to rotate
        v2 ((...,3) np.array): Desired vector

    """
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    return about_axis(_vector_bisector(v1, v2), np.pi)

def from_euler(alpha, beta, gamma):
    R"""Convert Euler angles to quaternion (3-2-1 convention)

    Args:
        alpha (float, np.array): First angle (in radians). May be an array of angles
        beta (float, np.array): Second angle (in radians). May be an array of angles
        gamma (float, np.array): Third angle (in radians). May be an array of angles

    Returns:
        An Nx4 np.array containing the products of row i of qi with column j of qj

    If the angles provided are arrays, their shapes must all match.
    Standard numpy broadcasting is used to compute the quaternions
    along the last dimension of the angle arrays.

    Note:
        Derived from injavis implementation

    Example::

        rands = np.random.rand(100, 3)
        alpha, beta, gamma = rands.T
        ql.from_euler(alpha, beta, gamma)
    """
    alpha = np.asarray(alpha)
    beta = np.asarray(beta)
    gamma = np.asarray(gamma)

    if any(type(x) == np.ndarray for x in (alpha, beta, gamma)):
        if not all(type(x) == np.ndarray for x in (alpha, beta, gamma)):
            raise ValueError("Either all inputs must be arrays, or none of them can be.")
        if not alpha.shape == beta.shape == gamma.shape:
            raise ValueError("All input arrays must be the same shape.")

    alpha = alpha * 0.5
    beta = beta * 0.5
    gamma = gamma * 0.5


    c1, c2, c3 = np.cos((alpha, beta, gamma))
    s1, s2, s3 = np.sin((alpha, beta, gamma))

    r = c1 * c2 * c3 + s1 * s2 * s3
    i = s1 * c2 * c3 - c1 * s2 * s3
    j = c1 * s2 * c3 + s1 * c2 * s3
    k = c1 * c2 * s3 - s1 * s2 * c3

    return np.stack([np.atleast_1d(x) for x in (r, i, j, k)], axis = -1)

def to_euler(q):
    R"""Convert quaternions to Euler angles (3-2-1 convention)

    Args:
        q ((...,4) np.array): Quaternions to transform

    Returns:
        A tuple of arrays (alpha, beta, gamma) with Euler angles in radians

    Note:
        Derived from injavis implementation

    Example::

        rands = np.random.rand(100, 3)
        alpha, beta, gamma = rands.T
        ql.from_euler(alpha, beta, gamma)
        alpha_return, beta_return, gamma_return = ql.to_euler(full)

    """
    q = np.asarray(q)

    r = q[..., 0]
    i = q[..., 1]
    j = q[..., 2]
    k = q[..., 3]

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

    alpha = np.atleast_1d(np.arctan2(2.0 * (q01 + q23), q00 - q11 - q22 + q33))
    beta = np.atleast_1d(np.arcsin(2.0 * (q02 - q13)))
    gamma = np.atleast_1d(np.arctan2(2.0 * (q03 + q12), q00 + q11 - q22 - q33))

    alpha[np.isnan(alpha)] = np.pi/2
    beta[np.isnan(beta)] = np.pi/2
    gamma[np.isnan(gamma)] = np.pi/2

    return (alpha, beta, gamma)

def from_matrix(mat, require_orthogonal = True):
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
        warnings.warn("Not all of your matrices are orthogonal. Please ensure that there are no improper rotations. If this was intentional, please set require_orthogonal to False when calling this function.", UserWarning)

    K = np.zeros(mat.shape[:-2]+(4, 4))
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
    K = K/3.0

    w, v = np.linalg.eigh(K)
    # The conventions in the paper are very confusing for quaternions in terms of the order of the components
    return np.concatenate((v[..., -1, -1, np.newaxis], -v[..., :-1, -1]), axis = -1)

def to_matrix(q, require_unit = True):
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
        raise ZeroDivisionError("At least one element of q has approximately zero norm")
    else:
        if not np.allclose(s, 1.0):
            warnings.warn("Not all quaternions in q are unit quaternions. If this was intentional, please set require_unit to False when calling this function.", UserWarning)
        m = np.empty(q.shape[:-1] + (3, 3))
        s **=-1.0 # For consistency with Wikipedia notation
        m[..., 0, 0] = 1.0 - 2*s*(q[..., 2]**2 + q[..., 3]**2)
        m[..., 0, 1] = 2*(q[..., 1]*q[..., 2] - q[..., 3]*q[..., 0])
        m[..., 0, 2] = 2*(q[..., 1]*q[..., 3] + q[..., 2]*q[..., 0])
        m[..., 1, 0] = 2*(q[..., 1]*q[..., 2] + q[..., 3]*q[..., 0])
        m[..., 1, 1] = 1.0 - 2*(q[..., 1]**2 + q[..., 3]**2)
        m[..., 1, 2] = 2*(q[..., 2]*q[..., 3] - q[..., 1]*q[..., 0])
        m[..., 2, 0] = 2*(q[..., 1]*q[..., 3] - q[..., 2]*q[..., 0])
        m[..., 2, 1] = 2*(q[..., 2]*q[..., 3] + q[..., 1]*q[..., 0])
        m[..., 2, 2] = 1.0 - 2*(q[..., 1]**2 + q[..., 2]**2)
        return m
