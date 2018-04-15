# Copyright (c) 2018 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
R"""
The general space of problems that this subpackage addresses is a small subset of the broader space of `point set registration <https://en.wikipedia.org/wiki/Point_set_registration>`_, which attempts to optimally align two sets of points. In general, this mapping can be nonlinear. The restriction of this superposition to linear transformations composed of translation, rotation, and scaling is the study of Procrustes superposition, the first step in the field of `Procrustes analysis <https://en.wikipedia.org/wiki/Procrustes_analysis#Shape_comparison>`_, which performs the superposition in order to compare two (or more) shapes.

There are numerous methods for solving this problem, most of which involve first centering both point sets on their centroids and then searching for the orthogonal transformation mapping between them. The determination of this transformation is known as the `orthogonal Procrustes problem <https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem>`_, and numerous algorithms have been developed to solve it.

Mathematically, the orthogonal Procrustes problem can be formulated as

.. math::
    \begin{equation}
        R = \argmin_\Omega \lvert\lvert\Omega A - B\rvert\rvert_F,\,\,
        \Omega^T\Omega = \mathbb{I}
    \begin{equation}

Note that the relevant norm here is the Frobenius matrix norm.

A closely related problem in attitude determination and aeronautics is Wahba's problem

.. math::
    \begin{equation}
        \min_{\boldsymbol{R} \in SO(3)} \frac{1}{2} \sum_{k=1}^N a_k \lvert \lvert
        \boldsymbol{w}_k - \boldsymbol{R} \boldsymbol{v}_k \rvert\rvert^2
    \begin{equation}

This problem can be seen as an alternative formulation of the orthogonal Procrustes problem by multiplying the weights :math:`a_k` into the vectors in the inner product and then considering the vectors as columns of matrices with :math:`N` entries. This formulation imposes the additional restriction that the mapping be a pure rotation (without reflection).

Many papers have been written reviewing the various solutions to the orthogonal Procrustes problem, most of which have been independently rediscovered by multiple authors in different fields. Of particular note are `Green's solution for full rank matrices <https://link.springer.com/article/10.1007/BF02288918>`_ and `Peter Schonemann's solution <https://link.springer.com/article/10.1007/BF02289451>_` to the orthogonal Procrustes problem (the first widely known solution, although alternatives were discovered earlier by von Neumann and `Fan and Hoffman <http://www.ams.org/journals/proc/1955-006-01/S0002-9939-1955-0067841-7/S0002-9939-1955-0067841-7.pdf>`). `This paper <https://ia800502.us.archive.org/16/items/nasa_techdoc_20000034107/20000034107.pdf>_` and `this document <https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19990104598.pdf>`_, both by Markley, provide overviews of numerous subsequent methods developed for solving Wahba's problem largely independent of the Procrustes solutions.

In the field of crystallography, a solution for the equivalent problem of minimizing the RMSD between two molecules is the `Kabsch algorithm <http://scripts.iucr.org/cgi-bin/paper?S0567739476001873>`_, which is commonly used in molecular modeling (and was also rediscovered by `Markley <https://www.researchgate.net/publication/243753921_Attitude_Determination_Using_Vector_Observations_and_the_Singular_Value_Decomposition>`_. The most popular and stable algorithms for solving Wahba's problem in other fields are variants on the `Davenport q-method <https://ntrs.nasa.gov/search.jsp?R=19670009376>`_, which is quaternion-based and involves computing eigenvalues of a particular 4x4 matrix constructed from the data.

This subpackage focuses specifically on providing algorithms to solve Wahba's problem. Since :py:module:`rowan` is a quaternion focused packaged, preference is given to implementing quaternion-based methods, although the Kabsch algorithm is included for completeness.


.. note:
    By nature of the problem being solved, this subpackage does not support
    any form of array broadcasting.
"""
from __future__ import division, print_function, absolute_import

import numpy as np

from ..functions import from_matrix, rotate

__all__ = ['kabsch']

def kabsch(p, q, require_rotation=True):
    R"""Find the optimal rotation and translation to map between two sets of
    points.

    This function implements the
    `Kabsch algorithm <https://en.wikipedia.org/wiki/Kabsch_algorithm>`, which
    minimizes the RMSD between two sets of points.

    Args:
        p ((N, 3) np.array): First set of N points
        q ((N, 3) np.array): Second set of N points
        require_rotation (bool): If false, the returned quaternion

    Returns:
        A tuple (q, t) where q is the quaternion to rotate the points and t
        is the tranlsation.
    """
    if p.shape != q.shape:
        raise ValueError("Input arrays must be the same shape")
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

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[2, :] *= -1
       R = Vt.T.dot(U.T)

    t = -R.dot(centroid_p.T) + centroid_q.T

    return from_matrix(R), t


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
        is the tranlsation.
    """
    if p.shape != q.shape:
        raise ValueError("Input arrays must be the same shape")
    p = np.asarray(p)
    q = np.asarray(q)

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
    <https://ntrs.nasa.gov/search.jsp?R=19670009376>`_, the most robust method and
    basis of most modern solvers. It involves the construction of a particular
    matrix, the Davenport K-matrix, which is then diagnolized to find the
    appropriate eigenvalues. More modern algorithms aim to solve the
    characteristic equation directly rather than diagonalizing, which can provide
    speed benefits at the potential cost of robustness.

    Args:
        p ((N, 3) np.array): First set of N points
        q ((N, 3) np.array): Second set of N points

    Returns:
        A tuple (q, t) where q is the quaternion to rotate the points and t
        is the tranlsation.
    """
    if p.shape != q.shape:
        raise ValueError("Input arrays must be the same shape")
    p = np.asarray(p)
    q = np.asarray(q)

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
    # quaternion with scalar part in the 4th entry. The q-matrix here is slightly
    # modified to avoid this problem
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
