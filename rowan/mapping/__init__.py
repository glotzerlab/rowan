# Copyright (c) 2018 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
R"""
The general space of problems that this subpackage addresses is a small subset
of the broader space of `point set registration
<https://en.wikipedia.org/wiki/Point_set_registration>`_, which attempts to
optimally align two sets of points. In general, this mapping can be nonlinear.
The restriction of this superposition to linear transformations composed of
translation, rotation, and scaling is the study of Procrustes superposition,
the first step in the field of `Procrustes analysis
<https://en.wikipedia.org/wiki/Procrustes_analysis#Shape_comparison>`_, which
performs the superposition in order to compare two (or more) shapes.

If points in the two sets have a known correspondence, the problem is much
simpler. Various precise formulations exist that admit analytical formulations,
such as the `orthogonal Procrustes problem
<https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem>`_ searching for an
orthogonal transformation

.. math::
    \begin{equation}
        R = \argmin_\Omega \lvert\lvert\Omega A - B\rvert\rvert_F,\,\,
        \Omega^T\Omega = \mathbb{I}
    \begin{equation}

or, if a pure rotation is desired, Wahba's problem

.. math::
    \begin{equation}
        \min_{\boldsymbol{R} \in SO(3)} \frac{1}{2} \sum_{k=1}^N a_k \lvert
        \lvert \boldsymbol{w}_k - \boldsymbol{R} \boldsymbol{v}_k \rvert\rvert^2
    \begin{equation}

Numerous algorithms to solve this problem exist, particularly in the field of
aerospace engineering and robotics where this problem must be solved on embedded
systems with limited processing. Since that constraint does not apply here, this
package simply implements some of the most stable known methods irrespective of
cost. In particular, this package contains the `Kabsch algorithm
<http://scripts.iucr.org/cgi-bin/paper?S0567739476001873>`_, which solves
Wahba's problem using an SVD in the vein of `Peter Schonemann's original
solution <https://link.springer.com/article/10.1007/BF02289451>_` to
the orthogonal Procrustes problem. Additionally this package contains the
`Davenport q method <https://ntrs.nasa.gov/search.jsp?R=19670009376>`_, which
works directly with quaternions. The most popular algorithms for Wahba's problem
are variants of the q method that are faster at the cost of some stability; we
omit these here.

In addition, :py:module:`rowan.mapping` also includes some functionality for
more general point set registration. If a point cloud has a set of known
symmetries, these can be tested explicitly by :py:module:`rowan.mapping` to
find the smallest rotation required for optimal mapping. If no such
correspondence is knowna at all, then the iterative closest point algorithm can
be used to approximate the mapping.
"""
from __future__ import division, print_function, absolute_import

import numpy as np

try:
    from sklearn.neighbors import NearestNeighbors
except ImportError:
    class NearestNeighbors(object):
        """Provide erros"""
        def __init__(self):
            raise ImportError("This method requires scikit-learn. Please "
                              "install sklearn and try again")

from ..functions import from_matrix, rotate

__all__ = ['kabsch', 'davenport', 'procrustes', 'icp']


def kabsch(p, q, require_rotation=True):
    R"""Find the optimal rotation and translation to map between two sets of
    points.

    This function implements the
    `Kabsch algorithm <https://en.wikipedia.org/wiki/Kabsch_algorithm>`, which
    minimizes the RMSD between two sets of points. One benefit of this approach
    is that the SVD works in dimensions > 3.

    Args:
        p ((N, m) np.array): First set of N points
        q ((N, m) np.array): Second set of N points
        require_rotation (bool): If false, the returned quaternion

    Returns:
        A tuple (R, t) where R is the (mxm) rotation matrix to rotate the
        points and t is the translation.
    """
    p = np.atleast_2d(p)
    q = np.atleast_2d(q)
    if p.shape != q.shape:
        raise ValueError("Input arrays must be the same shape")
    elif len(p.shape) != 2:
        raise ValueError("Input arrays must be (Nxm) arrays")
    p = np.asarray(p)
    q = np.asarray(q)

    # The algorithm depends on removing the centroid of the points.
    centroid_p = np.mean(p, axis=0)
    centroid_q = np.mean(q, axis=0)
    p_c = p - centroid_p
    q_c = q - centroid_q

    H = p_c.T.dot(q_c)

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T.dot(U.T)

    # Remove reflections by negating final eigenvalue
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T.dot(U.T)

    t = -R.dot(centroid_p.T) + centroid_q.T

    return R, t


def horn(p, q):
    R"""Find the optimal rotation and translation to map between two sets of
    points.

    This function implements the quaternion-based method of `Horn
    <https://www.osapublishing.org/josaa/abstract.cfm?id=2711>`_. For a simpler
    explanation, see `here
    <https://cnx.org/contents/HV-RsdwL@23/Molecular-Distance-Measures>`_.

    Args:
        p ((N, 3) np.array): First set of N points
        q ((N, 3) np.array): Second set of N points

    Returns:
        A tuple (q, t) where q is the quaternion to rotate the points and t
        is the translation.
    """
    p = np.atleast_2d(p)
    q = np.atleast_2d(q)
    if p.shape != q.shape:
        raise ValueError("Input arrays must be the same shape")
    elif len(p.shape) != 2 or p.shape[1] != 3:
        raise ValueError("Input arrays must be (Nx3) arrays")

    # The algorithm depends on removing the centroid of the points.
    centroid_p = np.mean(p, axis=0)
    centroid_q = np.mean(q, axis=0)
    p_c = p - centroid_p
    q_c = q - centroid_q

    A = np.empty((p.shape[0], 4, 4))
    A[:, 0, 0] = 0
    A[:, 1, 1] = 0
    A[:, 2, 2] = 0
    A[:, 3, 3] = 0
    A[:, [0, 3], [1, 2]] = -p_c[:, [0]]
    A[:, [1, 2], [0, 3]] = p_c[:, [0]]
    A[:, [0, 1], [2, 3]] = -p_c[:, [1]]
    A[:, [2, 3], [0, 1]] = p_c[:, [1]]
    A[:, [0, 2], [3, 1]] = -p_c[:, [2]]
    A[:, [1, 3], [2, 0]] = p_c[:, [2]]

    B = np.empty((q.shape[0], 4, 4))
    B[:, 0, 0] = 0
    B[:, 1, 1] = 0
    B[:, 2, 2] = 0
    B[:, 3, 3] = 0
    B[:, [0, 2], [1, 3]] = -q_c[:, [0]]
    B[:, [1, 3], [0, 2]] = q_c[:, [0]]
    B[:, [0, 3], [2, 1]] = -q_c[:, [1]]
    B[:, [2, 1], [0, 3]] = q_c[:, [1]]
    B[:, [0, 1], [3, 2]] = -q_c[:, [2]]
    B[:, [2, 3], [1, 0]] = q_c[:, [2]]

    prods = np.matmul(A.transpose(0, 2, 1), B)
    N = np.sum(prods, axis=0)

    # Note that Horn advocates solving the characteristic polynomial
    # explicitly to avoid computing an eigendecomposition; we do so
    # for simplicity
    w, v = np.linalg.eig(N)
    q_R = v[:, np.argmax(w)]

    t = -rotate(q_R, centroid_p) + centroid_q

    return q_R, t


def davenport(p, q):
    R"""Find the optimal rotation and translation to map between two sets of
    points.

    This function implements the `Davenport q-method
    <https://ntrs.nasa.gov/search.jsp?R=19670009376>`_, the most robust method
    and basis of most modern solvers. It involves the construction of a
    particular matrix, the Davenport K-matrix, which is then diagnolized to find
    the appropriate eigenvalues. More modern algorithms aim to solve the
    characteristic equation directly rather than diagonalizing, which can
    provide speed benefits at the potential cost of robustness.

    Args:
        p ((N, 3) np.array): First set of N points
        q ((N, 3) np.array): Second set of N points

    Returns:
        A tuple (q, t) where q is the quaternion to rotate the points and t
        is the translation.
    """
    p = np.atleast_2d(p)
    q = np.atleast_2d(q)
    if p.shape != q.shape:
        raise ValueError("Input arrays must be the same shape")
    elif len(p.shape) != 2 or p.shape[1] != 3:
        raise ValueError("Input arrays must be (Nx3) arrays")

    # The algorithm depends on removing the centroid of the points.
    centroid_p = np.mean(p, axis=0)
    centroid_q = np.mean(q, axis=0)
    p_c = p - centroid_p
    q_c = q - centroid_q

    B = p_c.T.dot(q_c)
    tr_B = np.trace(B)
    z = np.array([B[1, 2] - B[2, 1], B[2, 0] - B[0, 2], B[0, 1] - B[1, 0]]).T

    # Note that the original Davenport q-matrix puts the solution to the vector
    # part of the quaternion in the upper right block; this results in a
    # quaternion with scalar part in the 4th entry. The q-matrix here is
    # slightly modified to avoid this problem
    K = np.empty((4, 4))
    K[1:, 1:] = B + B.T - np.eye(3)*tr_B
    K[0, 1:] = z.T
    K[1:, 0] = z
    K[0, 0] = tr_B
#    K[:3, :3] = B + B.T - np.eye(3)*tr_B
#    K[3, :3] = z.T
#    K[:3, 3] = z
#    K[3, 3] = tr_B

    w, v = np.linalg.eig(K)
    q_R = v[:, np.argmax(w)]

    t = -rotate(q_R, centroid_p) + centroid_q

    return q_R, t


# TODO: Allow specification of equivalent orientations
def procrustes(p, q, method='best'):
    R"""Solve the orthogonal Procrustes problem.

    This function provides an interface to multiple algorithms to
    solve the orthogonal Procrustes problem.

    Args:
        p ((N, m) np.array): First set of N points
        q ((N, m) np.array): Second set of N points
        method (str): A method to use. Options are 'kabsch', 'davenport'
            and 'horn'. The default is to select the best option ('best')

    Returns:
        A tuple (q, t) where q is the quaternion to rotate the points and t
        is the translation.
    """
    import sys
    thismodule = sys.modules[__name__]

    if method != 'best':
        try:
            method = getattr(thismodule, method)
        except KeyError:
            raise ValueError("The input method is not known")
    else:
        p = np.atleast_2d(p)
        q = np.atleast_2d(q)
        if p.shape != q.shape:
            raise ValueError("Input arrays must be the same shape")
        elif len(p.shape) != 2:
            raise ValueError("Input arrays must be 2d arrays")
        if p.shape[1] != 3:
            method = getattr(thismodule, 'kabsch')
        else:
            method = getattr(thismodule, 'davenport')
    return method(p, q)


def icp(A, B, method='best', unique_match=True, max_iterations=20, tolerance=0.001):
    '''
    Apply the Iterative Closest Point algorithm to find the optimal mapping.
    Args:
        A ((N, m) np.array): First set of N points
        B ((N, m) np.array): Second set of N points
        method (str): A method to use for each alignment. Options are 'kabsch',
            'davenport' and 'horn'. The default is to select the best option
            ('best').
        unique_match (bool): Whether to require nearest neighbors to be unique.
        max_iterations (int): Number of iterations to attempt.
        tolerance (float): Indicates convergence

    Returns:
        A tuple (R, t) where q is the quaternion to rotate the points and t
        is the translation.
    '''

    import sys
    thismodule = sys.modules[__name__]

    if method != 'best':
        try:
            method = getattr(thismodule, method)
        except KeyError:
            raise ValueError("The input method is not known")
    else:
        A = np.atleast_2d(A)
        B = np.atleast_2d(B)
        if A.shape != B.shape:
            raise ValueError("Input arrays must be the same shape")
        elif len(A.shape) != 2:
            raise ValueError("Input arrays must be (Nx3) arrays")
        if A.shape[1] != 3:
            method = getattr(thismodule, 'kabsch')
        else:
            method = getattr(thismodule, 'davenport')

    # make points homogeneous, copy them to maintain the originals
    src = np.copy(A)
    dst = np.copy(B)

    prev_error = 0

    if unique_match:
        try:
            from scipy import spatial, optimize
        except ImportError:
            raise ImportError("Running with unique_match requires "
                              " scipy. Please install sklearn and try "
                              " again.")
    else:
        try:
            from sklearn import neighbors
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(dst)
        except ImportError:
            raise ImportError("Running without unique_match requires "
                              "scikit-learn. Please install sklearn and try "
                              "again.")

    for i in range(max_iterations):
        # Rather than a coarse nearest neighbors, we apply the Hungarian
        # algorithm to ensure that we do not have duplicates. Unfortunately,
        # this precludes acceleration of the spatial search but is worthwhile
        # for the improved accuracy
        if unique_match:
            pair_distances = spatial.distance.cdist(src, dst)
            row_ind, indices = optimize.linear_sum_assignment(pair_distances)
            distances = pair_distances[row_ind, indices]
        else:
            distances, indices = nn.kneighbors(src, return_distance=True)
            distances = distances.ravel()
            indices = indices.ravel()

        # compute the transformation between the current source and nearest destination points
        q, t = method(src, dst[indices, :])

        # update the current source
        if q.shape[-1] != 4:
            # Returned a matrix instead of a quaternion
            src = np.dot(src, q.T) + t
        else:
            src = rotate(q, src) + t

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    q, t = method(A, src)

    if q.shape[-1] == 4:
        R = to_matrix(q)

    return R, t
