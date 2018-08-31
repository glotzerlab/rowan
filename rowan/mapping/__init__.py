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
        R = \textrm{argmin}_\Omega \lvert\lvert\Omega A - B\rvert\rvert_F,\,\,
        \Omega^T\Omega = \mathbb{1}
    \end{equation}

or, if a pure rotation is desired, Wahba's problem

.. math::
    \begin{equation}
        \min_{\boldsymbol{R} \in SO(3)} \frac{1}{2} \sum_{k=1}^N a_k \lvert
        \lvert \boldsymbol{w}_k - \boldsymbol{R} \boldsymbol{v}_k \rvert\rvert^2
    \end{equation}

Numerous algorithms to solve this problem exist, particularly in the field of
aerospace engineering and robotics where this problem must be solved on embedded
systems with limited processing. Since that constraint does not apply here, this
package simply implements some of the most stable known methods irrespective of
cost. In particular, this package contains the Kabsch algorithm, which solves
Wahba's problem using an SVD in the vein of `Peter Schonemann's original
solution <https://link.springer.com/article/10.1007/BF02289451>`_ to
the orthogonal Procrustes problem. Additionally this package contains the
`Davenport q method <https://ntrs.nasa.gov/search.jsp?R=19670009376>`_, which
works directly with quaternions. The most popular algorithms for Wahba's problem
are variants of the q method that are faster at the cost of some stability; we
omit these here.

In addition, :py:mod:`rowan.mapping` also includes some functionality for
more general point set registration. If a point cloud has a set of known
symmetries, these can be tested explicitly by :py:mod:`rowan.mapping` to
find the smallest rotation required for optimal mapping. If no such
correspondence is knowna at all, then the iterative closest point algorithm can
be used to approximate the mapping.
"""
from __future__ import division, print_function, absolute_import

import numpy as np

from ..functions import from_matrix, rotate, to_matrix
from ..geometry import angle

__all__ = ['kabsch', 'davenport', 'procrustes', 'icp']


def kabsch(X, Y, require_rotation=True):
    R"""Find the optimal rotation and translation to map between two sets of
    points.

    This function implements the
    `Kabsch algorithm <https://en.wikipedia.org/wiki/Kabsch_algorithm>`_, which
    minimizes the RMSD between two sets of points. One benefit of this approach
    is that the SVD works in dimensions > 3.

    Args:
        X ((N, m) np.array): First set of N points.
        Y ((N, m) np.array): Second set of N points.
        require_rotation (bool): If false, the returned quaternion.

    Returns:
        A tuple (R, t) where R is the (m x m) rotation matrix to rotate the
        points and t is the translation.

    Example::

        import numpy as np

        # Create some random points, then make a random transformation of
        # these points
        points = np.random.rand(10, 3)
        rotation = rowan.random.rand(1)
        translation = np.random.rand(1, 3)
        transformed_points = rowan.rotate(rotation, points) + translation

        # Recover the rotation and check
        R, t = rowan.mapping.kabsch(points, transformed_points)
        q = rowan.from_matrix(R)

        assert np.logical_or(
            np.allclose(rotation, q), np.allclose(rotation, -q))
        assert np.allclose(translation, t)
    """
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    if X.shape != Y.shape:
        raise ValueError("Input arrays must be the same shape")
    elif len(X.shape) != 2:
        raise ValueError("Input arrays must be (Nxm) arrays")
    X = np.asarray(X)
    Y = np.asarray(Y)

    # The algorithm depends on removing the centroid of the points.
    centroid_X = np.mean(X, axis=0)
    centroid_Y = np.mean(Y, axis=0)
    X_c = X - centroid_X
    Y_c = Y - centroid_Y

    H = X_c.T.dot(Y_c)

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T.dot(U.T)

    # Remove reflections by negating final eigenvalue
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T.dot(U.T)

    t = -R.dot(centroid_X.T) + centroid_Y.T

    return R, t


def horn(X, Y):
    R"""Find the optimal rotation and translation to map between two sets of
    points.

    This function implements the quaternion-based method of `Horn
    <https://www.osapublishing.org/josaa/abstract.cfm?id=2711>`_. For a simpler
    explanation, see `here
    <https://cnx.org/contents/HV-RsdwL@23/Molecular-Distance-Measures>`_.

    Args:
        X ((N, 3) np.array): First set of N points.
        Y ((N, 3) np.array): Second set of N points.

    Returns:
        A tuple (q, t) where q is the quaternion to rotate the points and t
        is the translation.

    Example::

        import numpy as np

        # Create some random points, then make a random transformation of
        # these points
        points = np.random.rand(10, 3)
        rotation = rowan.random.rand(1)
        translation = np.random.rand(1, 3)
        transformed_points = rowan.rotate(rotation, points) + translation

        # Recover the rotation and check
        q, t = rowan.mapping.horn(points, transformed_points)

        assert np.logical_or(
            np.allclose(rotation, q), np.allclose(rotation, -q))
        assert np.allclose(translation, t)
    """
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    if X.shape != Y.shape:
        raise ValueError("Input arrays must be the same shape")
    elif len(X.shape) != 2 or X.shape[1] != 3:
        raise ValueError("Input arrays must be (Nx3) arrays")

    # The algorithm depends on removing the centroid of the points.
    centroid_X = np.mean(X, axis=0)
    centroid_Y = np.mean(Y, axis=0)
    X_c = X - centroid_X
    Y_c = Y - centroid_Y

    A = np.empty((X.shape[0], 4, 4))
    A[:, 0, 0] = 0
    A[:, 1, 1] = 0
    A[:, 2, 2] = 0
    A[:, 3, 3] = 0
    A[:, [0, 3], [1, 2]] = -X_c[:, [0]]
    A[:, [1, 2], [0, 3]] = X_c[:, [0]]
    A[:, [0, 1], [2, 3]] = -X_c[:, [1]]
    A[:, [2, 3], [0, 1]] = X_c[:, [1]]
    A[:, [0, 2], [3, 1]] = -X_c[:, [2]]
    A[:, [1, 3], [2, 0]] = X_c[:, [2]]

    B = np.empty((Y.shape[0], 4, 4))
    B[:, 0, 0] = 0
    B[:, 1, 1] = 0
    B[:, 2, 2] = 0
    B[:, 3, 3] = 0
    B[:, [0, 2], [1, 3]] = -Y_c[:, [0]]
    B[:, [1, 3], [0, 2]] = Y_c[:, [0]]
    B[:, [0, 3], [2, 1]] = -Y_c[:, [1]]
    B[:, [2, 1], [0, 3]] = Y_c[:, [1]]
    B[:, [0, 1], [3, 2]] = -Y_c[:, [2]]
    B[:, [2, 3], [1, 0]] = Y_c[:, [2]]

    prods = np.matmul(A.transpose(0, 2, 1), B)
    N = np.sum(prods, axis=0)

    # Note that Horn advocates solving the characteristic polynomial
    # explicitly to avoid computing an eigendecomposition; we do so
    # for simplicity
    w, v = np.linalg.eig(N)
    q = v[:, np.argmax(w)]

    t = -rotate(q, centroid_X) + centroid_Y

    return q, t


def davenport(X, Y):
    R"""Find the optimal rotation and translation to map between two sets of
    points.

    This function implements the `Davenport q-method
    <https://ntrs.nasa.gov/search.jsp?R=19670009376>`_, the most robust method
    and basis of most modern solvers. It involves the construction of a
    particular matrix, the Davenport K-matrix, which is then diagonalized to
    find the appropriate eigenvalues. More modern algorithms aim to solve the
    characteristic equation directly rather than diagonalizing, which can
    provide speed benefits at the potential cost of robustness. The
    implementation in ``rowan`` does not do this, instead simply computing the
    spectral decomposition.

    Args:
        X ((N, 3) np.array): First set of N points.
        Y ((N, 3) np.array): Second set of N points.

    Returns:
        A tuple (q, t) where q is the quaternion to rotate the points and t
        is the translation.

    Example::

        import numpy as np

        # Create some random points, then make a random transformation of
        # these points
        points = np.random.rand(10, 3)
        rotation = rowan.random.rand(1)
        translation = np.random.rand(1, 3)
        transformed_points = rowan.rotate(rotation, points) + translation

        # Recover the rotation and check
        q, t = rowan.mapping.davenport(points, transformed_points)

        assert np.logical_or(
            np.allclose(rotation, q), np.allclose(rotation, -q))
        assert np.allclose(translation, t)
    """
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    if X.shape != Y.shape:
        raise ValueError("Input arrays must be the same shape")
    elif len(X.shape) != 2 or X.shape[1] != 3:
        raise ValueError("Input arrays must be (Nx3) arrays")

    # The algorithm depends on removing the centroid of the points.
    centroid_X = np.mean(X, axis=0)
    centroid_Y = np.mean(Y, axis=0)
    X_c = X - centroid_X
    Y_c = Y - centroid_Y

    B = X_c.T.dot(Y_c)
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

    w, v = np.linalg.eig(K)
    q = v[:, np.argmax(w)]

    t = -rotate(q, centroid_X) + centroid_Y

    return q, t


def procrustes(X, Y, method='best', equivalent_quaternions=None):
    R"""Solve the orthogonal Procrustes problem with algorithmic options.

    Args:
        X ((N, m) np.array): First set of N points.
        Y ((N, m) np.array): Second set of N points.
        method (str): A method to use. Options are 'kabsch', 'davenport'
            and 'horn'. The default is to select the best option ('best').
        equivalent_quaternions (array-like): If the precise correspondence is
            not known, but the points are known to be part of a body with
            specific symmetries, the set of quaternions generating
            symmetry-equivalent configurations can be provided. These
            quaternions will be tested exhaustively to find the smallest
            symmetry-equivalent rotation.

    Returns:
        A tuple (q, t) where q is the quaternion to rotate the points and t
        is the translation.

    Example::

        import numpy as np

        # Create some random points, then make a random transformation of
        # these points
        points = np.random.rand(10, 3)
        rotation = rowan.random.rand(1)
        translation = np.random.rand(1, 3)
        transformed_points = rowan.rotate(rotation, points) + translation

        # Recover the rotation and check
        q, t = rowan.mapping.procrustes(
            points, transformed_points, method='horn')

        assert np.logical_or(
            np.allclose(rotation, q), np.allclose(rotation, -q))
        assert np.allclose(translation, t)
    """
    import sys
    thismodule = sys.modules[__name__]

    if method != 'best':
        try:
            method = getattr(thismodule, method)
        except KeyError:
            raise ValueError("The input method is not known")
    else:
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        if X.shape != Y.shape:
            raise ValueError("Input arrays must be the same shape")
        elif len(X.shape) != 2:
            raise ValueError("Input arrays must be 2D arrays")
        if X.shape[1] != 3:
            method = getattr(thismodule, 'kabsch')
        else:
            method = getattr(thismodule, 'davenport')
    if equivalent_quaternions is not None:
        qs = []
        ts = []
        for eq in equivalent_quaternions:
            q, t = method(rotate(eq, X), Y)
            if method.__name__ == 'kabsch':
                qs.append(from_matrix(q))
            else:
                qs.append(q)
            ts.append(t)
        index = np.argmin([angle(q) for q in qs])
        return qs[index], ts[index]
    else:
        q, t = method(X, Y)
        if method == 'kabsch':
            return from_matrix(q), t
        else:
            return q, t


def icp(X, Y, method='best', unique_match=True, max_iterations=20,
        tolerance=0.001):
    R"""Find best mapping using the Iterative Closest Point algorithm.

    Args:
        X ((N, m) np.array): First set of N points.
        Y ((N, m) np.array): Second set of N points.
        method (str): A method to use for each alignment. Options are 'kabsch',
            'davenport' and 'horn'. The default is to select the best option
            ('best').
        unique_match (bool): Whether to require nearest neighbors to be unique.
        max_iterations (int): Number of iterations to attempt.
        tolerance (float): Indicates convergence.

    Returns:
        A tuple (R, t) where R is the matrix to rotate the points and t
        is the translation.

    Example::

        import numpy as np

        # Create some random points, then make a random transformation of
        # these points
        points = np.random.rand(10, 3)

        # Only works for small rotations
        rotation = rowan.from_axis_angle((1, 0, 0), 0.01)
        translation = np.random.rand(1, 3)
        transformed_points = rowan.rotate(rotation, points) + translation

        # Recover the rotation and check
        R, t = rowan.mapping.icp(points, transformed_points)
        q = rowan.from_matrix(R)

        assert np.logical_or(
            np.allclose(rotation, q), np.allclose(rotation, -q))
        assert np.allclose(translation, t)
    """

    import sys
    thismodule = sys.modules[__name__]

    if method != 'best':
        try:
            method = getattr(thismodule, method)
        except KeyError:
            raise ValueError("The input method is not known")
    else:
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        if X.shape != Y.shape:
            raise ValueError("Input arrays must be the same shape")
        elif len(X.shape) != 2:
            raise ValueError("Input arrays must be (Nx3) arrays")
        if X.shape[1] != 3:
            method = getattr(thismodule, 'kabsch')
        else:
            method = getattr(thismodule, 'davenport')

    if unique_match:
        try:
            from scipy import spatial, optimize
        except ImportError:
            raise ImportError("Running with unique_match requires scipy. "
                              "Please install scipy and try again.")
    else:
        try:
            from sklearn import neighbors
            nn = neighbors.NearestNeighbors(n_neighbors=1)
            nn.fit(Y)
        except ImportError:
            raise ImportError("Running without unique_match requires "
                              "scikit-learn. Please install sklearn and try "
                              "again.")

    # Copy points so we have originals available.
    cur_points = np.copy(X)
    err_old = 0
    for i in range(max_iterations):
        # Rather than a coarse nearest neighbors, we apply the Hungarian
        # algorithm to ensure that we do not have duplicates. Unfortunately,
        # this precludes acceleration of the spatial search but is worthwhile
        # for the improved accuracy
        if unique_match:
            pair_distances = spatial.distance.cdist(cur_points, Y)
            row_ind, indices = optimize.linear_sum_assignment(pair_distances)
            distances = pair_distances[row_ind, indices]
        else:
            distances, indices = nn.kneighbors(cur_points, return_distance=True)
            distances = distances.ravel()
            indices = indices.ravel()

        # Compute current best transformation
        q, t = method(cur_points, Y[indices, :])

        # Update the current source
        if q.shape[-1] != 4:
            # Returned a matrix instead of a quaternion
            cur_points = np.dot(cur_points, q.T) + t
        else:
            cur_points = rotate(q, cur_points) + t

        # Tolerance check
        err = np.mean(distances)
        if np.abs(err_old - err) < tolerance:
            break
        err_old = err

    # calculate final transformation
    q, t = method(X, cur_points)

    if q.shape[-1] == 4:
        R = to_matrix(q)

    return R, t
