"""Simple quaternion library containing standard methods"""

import numpy as np

__all__ = []

__version__ = 0.1

def quat_conjugate(q):
    R"""Returns the conjugate of quaternion array q

    Args:
        q (Nx4 np.array): First set of quaternions

    Returns:
        An Nx4 np.array containing the conjugates of q

    Example::

        q_star = quat_conjugate(q)
    """
    conjugate = q.copy()
    conjugate[:, 1:] *= -1
    return conjugate

def quat_rotate(q, v):
    R"""Rotates the vectors v by the quaternions q

    Args:
        q (Nx4 np.array): First set of quaternions
        v (Nx3 np.array): First set of quaternions

    Returns:
        An Nx3 np.array of the vectors in v rotated by q

    Example::

        q = np.random.rand(1, 4)
        v = np.random.rand(1, 3)
        v_rot = quat_rotate(q, v)
    """
    # Convert vector to quaternion representation
    v = _check_vec(v)
    q = _check_quat(q)
    quat_v = np.hstack((np.zeros((v.shape[0], 1)), v))
    return quat_multiply(q, quat_multiply(quat_v, quat_conjugate(q)))[:, 1:]

def quatrot(q, v):
    return ((q[:, 0]**2 - np.sum(q[:, 1:]**2, axis=1))[:, np.newaxis]*v +
            2*q[:, 0][:, np.newaxis]*np.cross(q[:, 1:], v) +
            2*np.sum(q[:, 1:]*v, axis=1)[:, np.newaxis]*q[:, 1:])

def _check_array(arr, dim = 4):
    R"""Check that arr conforms to the desired shape of input arrays

    Args:
        arr (np.array): Array of vectors or quaterions
        dim (int): 3 for vectors, 4 for quaternions

    Returns:
        Array with all dimensions fixed
    """
    if len(arr.shape) == 1:
        if not arr.shape[0] == dim:
            raise ValueError("Your input arrays are malformed! At least one dimensions must be of length {}.".format(dim))
        else:
            # Convert to 2D instead of 1D
            arr = arr[np.newaxis, :]
    elif len(arr.shape) == 2:
        if not arr.shape[1] == dim:
            raise ValueError("Your input arrays are malformed! The second dimension must be of length {}.".format(dim))
    else:
        raise ValueError("Your input arrays are malformed! They should be Nx{}.".format(dim))
    return arr

def _check_vec(v):
    R"""Check that v conforms to the desired shape of vector arrays

    Args:
        v (np.array): Array of vectors (hopefully Nx3)

    Returns:
        An Nx3 array of vectors
    """
    return _check_array(v, 3)

def _check_quat(q):
    R"""Check that q conforms to the desired shape of quaternion arrays

    Args:
        q (np.array): Array of quaterions (hopefully Nx4)

    Returns:
        An Nx4 array of quaternions that can be used throughout this module.
    """
    return _check_array(q, 4)

def quat_multiply(qi, qj):
    R"""Multiplies the quaternions in the array qi by those in qj

    Args:
        qi (Nx4 np.array): First set of quaternions
        qj (Nx4 np.array): Second set of quaternions

    Returns:
        An Nx4 np.array containing the products of row i of qi with column j of qj

    Example::

        qi = np.array([[1, 0, 0, 0]])
        qj = np.array([[1, 0, 0, 0]])
        prod = quat_multiply(qi, qj)
    """

    # Check for shapes of the arrays
    qi = _check_quat(qi)
    qj = _check_quat(qj)

    if not qi.shape == qj.shape:
        raise ValueError("The two arrays must be the same size!")

    output = np.empty(qi.shape)

    output[:, 0] = qi[:, 0] * qj[:, 0] - np.sum(qi[:, 1:] * qj[:, 1:], axis=1)
    output[:, 1:] = (qi[:, 0][:, np.newaxis] * qj[:, 1:] +
                     qj[:, 0][:, np.newaxis] * qi[:, 1:] +
                     np.cross(qi[:, 1:], qj[:, 1:]))
    return output

def normalize(q):
    R"""Get the normalized quaternion

    Args:
        q (Nx4 np.array): Quaternions to normalize

    Returns:
        An Nx4 np.array containing the unit quaternions qi/norm(qi)

    Example::

        q = np.random.rand(10, 4)
        u = normalize(q)
    """
    q = _check_quat(q)
    return q/np.linalg.norm(q, axis = 1)

def norm(q):
    R"""Trivial implementation of quaternion norm

    Returns:
        A length N np.array containing the norms for qi in q

    Example::

        q = np.random.rand(10, 4)
        norms = norm(q)
    """
    q = _check_quat(q)
    return np.linalg.norm(q)

def quaternion_rotate_about_axis(v, theta):
    R"""Find the quaternion corresponding to rotation about the axis v by angle theta

    Args:
        v (1x3 np.array): Axis to rotate about
        theta (float): Angle (in radians)

    Example::

        qi = np.array([[1, 0, 0, 0]])
        qj = np.array([[1, 0, 0, 0]])
        prod = quat_multiply(qi, qj)
    """

    # Now normalize
    u = np.normalize(v)
    ha = theta / 2.0
    q = np.zeros(4)
    q[0] = np.cos(ha)
    q[1:] = np.sin(ha) * u
    return q[np.newaxis, :]

def _vector_bisector(v1, v2):
    R"""Find the vector bisecting v1 and v2

    Args:
        v1 (1x3 np.array): First vector
        v2 (1x3 np.array): Second vector

    Returns:
        The vector that bisects the angle between v1 and v2

    """

    # Check that the vectors are reasonable
    if len(v1.shape) == 1:
        v1 = v1[np.newaxis, :]
    if len(v2.shape) == 2:
        v2 = v2[np.newaxis, :]
    if not v1.shape == v2.shape == [1, 3]:
        raise ValueError("The two inputs must both be 1x3 vectors")
    return normalize(normalize(v1) + normalize(v2))

def vector_vector_rotation(v1, v2):
    R"""Find the quaternion to rotate v1 onto v2

    Args:
        v1 (1x3 np.array): Vector to rotate
        v2 (1x3 np.array): Desired vector

    """
    return quaternion_about_axis(_vector_bisector(v1, v2), np.pi)

def quaternion_from_rotation_matrix(mat, require_orthogonal = True):
    R"""Convert the rotation mat to a quaternion

    Based on this paper <https://doi.org/10.2514/2.4654>. Conventions are sort of strange,
    so I'm not yet sure that everything works as expected.

    Args:
        mat (3x3 np.array): A rotation matrix

    Returns:
        The 1d quaternion representation of mat (i.e. the same element of SO(3))
    """
    if mat.shape != (3, 3):
        raise ValueError("The input must be a 3x3 matrix!")
    elif not np.allclose(np.linalg.norm(mat, axis = 1), 1) and require_orthogonal:
        raise ValueError("The input matrix must be orthogonal")

    K = np.zeros((4, 4))
    K[0, 0] = mat[0, 0] - mat[1, 1] - mat[2, 2]
    K[0, 1] = mat[1, 0] + mat[0, 1]
    K[0, 2] = mat[2, 0] + mat[0, 2]
    K[0, 3] = mat[1, 2] - mat[2, 1]
    K[1, 0] = mat[1, 0] + mat[0, 1]
    K[1, 1] = mat[1, 1] - mat[0, 0] - mat[2, 2]
    K[1, 2] = mat[2, 1] + mat[1, 2]
    K[1, 3] = mat[2, 0] - mat[0, 2]
    K[2, 0] = mat[2, 0] + mat[0, 2]
    K[2, 1] = mat[2, 1] + mat[1, 2]
    K[2, 2] = mat[2, 2] - mat[0, 0] - mat[1, 1]
    K[2, 3] = mat[0, 1] - mat[1, 0]
    K[3, 0] = mat[1, 2] - mat[2, 1]
    K[3, 1] = mat[2, 0] - mat[0, 2]
    K[3, 2] = mat[0, 1] - mat[1, 0]
    K[3, 3] = mat[0, 0] + mat[1, 1] + mat[2, 2]
    K = K/3.0

    w, v = np.linalg.eigh(K)
    # The conventions in the paper are very confusing for quaternions in terms of the order of the components
    return np.hstack((v[-1, -1], -v[:-1, -1]))
