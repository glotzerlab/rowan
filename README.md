# README

## About

The hamilton package provides a simple and consistent interface for using quaternions in code in the Glotzer Group University of Michigan, Ann Arbor.
The package is built entirely on top of numpy and represents quaternions using numpy arrays of dimension `...x4`, meaning that all functions support arbitrarily high-dimensional arrays of quaternions.

Quaternions are encoded as numpy arrays of length 4 with the convention that an array `(a, b, c, d)` represents the quaternion `a + bi + cj + dk`.
Almost all functions use entirely standard; the sole exception are the matrix-quaternion interconversions, which are more involved.
Matrices are converted to quaternions via the algorithm described by [Bar-Itzhack et al.](https://doi.org/10.2514/2.4654).
Quaternions are converted to matrices using the standard mathematical formula given on [Wikipedia's page on Quaternions and spatial rotation](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix).
Euler angle conversions are derived using standard rotation matrix representations.

## Authors

* Vyas Ramasubramani, vramasub@umich.edu (Maintainer)

## Setup

### Installation from source

To install from source, execute:

	git clone https://bitbucket.org/vramasub/hamilton.git
	cd hamilton
	python setup.py install --user

### Requirements

* Python = 3.6
* Numpy >= 1.7

## Testing

The package is currently tested for python version 3.6 on Unix.
Continuous integration testing is performed using CircleCI.

To run the packaged unit tests, execute:

    python -m unittest discover tests

## Quickstart
This library can be used to work with quaternions by simply instantiating the appropriate numpy arrays and passing them to the required functions.
For example:

    >>> import hamilton
    >>> one = np.array([10, 0, 0, 0])
    >>> one_unit = hamilton.normalize(one)
    >>> assert(one_unit == np.array([1, 0, 0, 0]))
    >>> if not one_unit == hamilton.quat_multiply(one_unit, one_unit):
    >>>     raise RuntimeError("Multiplication failed!")
    >>>
    >>> one_vec = np.array([1, 0, 0])
    >>> rotated_vector = hamilton.rotate(one_unit, one_vec)
    >>>
    >>> import numpy as np
    >>> mat = np.eye(3)
    >>> quat_rotate = hamilton.from_matrix(mat)
    >>> alpha, beta, gamma = hamilton.to_euler(quat_rotate)
    >>> quat_rotate_returned = hamilton.from_euler(alpha, beta, gamma)
    >>> identity = hamilton.to_matrix(quat_rotate_returned)
