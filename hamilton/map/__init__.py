# Copyright (c) 2018 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
"""Find optimal mappings between two sets of points.

`hamilton.map` provides methods for solving a number of closely related
problems involving finding optimal mappings between two sets of points.

Broadly speaking, the class of problems to solve is the following: given
two matrices matrix to find the 

The first instance of this classical problem is the `orthogonal Procrustes
problem <https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem>`,
while more recent related formulations include `Wahba's Problem
<https://en.wikipedia.org/wiki/Wahba%27s_problem>`. More general
formulations fall under the broad umbrella of generalized or constrained
Procrustes problems.

A number of different solutions have been developed to solve this problem
in different fields. Peter Schönemann's original solution to the orthogonal
Procrustes problem solves for 
"""

import numpy as np

__all__ = ['rand',
           ]

def rigid_transform_3D_array(A, B):
    """Determine the optimal rotation and translation to map one set of points
    onto another.

    This method uses the
    `Kabsch algorithm <https://en.wikipedia.org/wiki/Kabsch_algorithm>`, which
    is based on computing the SVD to minimize the covariance between the two
    datasets.

    Args:
        TBD

    Return:
        TBD
    """
    assert len(A) == len(B)

    N = A.shape[0]; # total points

    centroid_A = mean(A, axis=0)
    centroid_B = mean(B, axis=0)

    # centre the points
    AA = A - tile(centroid_A, (N, 1))
    BB = B - tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = AA.T.dot(BB)

    U, S, Vt = linalg.svd(H)

    R = Vt.T.dot(U.T)

    # special reflection case
    if linalg.det(R) < 0:
       print("Reflection detected")
       Vt[2,:] *= -1
       R = Vt.T.dot(U.T)

    t = -R.dot(centroid_A.T) + centroid_B.T

    #print(t)

    return R, t
