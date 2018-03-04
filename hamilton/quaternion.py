# Copyright (c) 2018 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
"""Define a quaternion class for users who desire a more encapsulated class."""

from .functions import (conjugate, multiply, norm, normalize,
                        rotate, about_axis, vector_vector_rotation,
                        from_euler, to_euler, from_matrix, to_matrix)


class quaternion:
    """Quaternion class encapsulating the natural operations.
    For now this only works for individual quaternions"""

    def __init__(self, array):
        if array.shape != (1, 4):
            raise ValueError(
                "Currently the quaternion class only supports construction using \
                    arrays of 4 elements")
        self.array = array

    def __repr__(self):
        return "quaternion(" + self.array.__repr__() + ")"

    def __str__(self, other):
        return self.array.__str__()

    def __eq__(self, other):
        return self.array == other.array

    def __add__(self, other):
        try:
            return self.__class__(self.array + other.array)
        except TypeError:
            raise TypeError(
                "Quaternions arithmetic only operates on two quaternions")
        else:
            raise

    def __sub__(self, other):
        try:
            return self.__class__(self.array - other.array)
        except TypeError:
            raise TypeError(
                "Quaternions arithmetic only operates on two quaternions")
        else:
            raise

    def __mul__(self, other):
        try:
            return self.__class__(multiply(self.array, other.array))
        except TypeError:
            raise TypeError(
                "Quaternions arithmetic only operates on two quaternions")
        else:
            raise

    def __truediv__(self, other):
        try:
            return self.__class__(multiply(self.array, other.conj))
        except TypeError:
            raise TypeError(
                "Quaternions arithmetic only operates on two quaternions")
        else:
            raise

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        try:
            return self.__class__(other.array - self.array)
        except TypeError:
            raise TypeError(
                "Quaternions arithmetic only operates on two quaternions")
        else:
            raise

    def __rmul__(self, other):
        try:
            return self.__class__(multiply(other.array, self.array))
        except TypeError:
            raise TypeError(
                "Quaternions arithmetic only operates on two quaternions")
        else:
            raise

    def __rtruediv__(self, other):
        try:
            return self.__class__(multiply(other.array, self.conj))
        except TypeError:
            raise TypeError(
                "Quaternions arithmetic only operates on two quaternions")
        else:
            raise

    def __iadd__(self, other):
        try:
            self.array += other.array
        except TypeError:
            raise TypeError(
                "Quaternions arithmetic only operates on two quaternions")
        else:
            raise

    def __isub__(self, other):
        try:
            self.array -= other.array
        except TypeError:
            raise TypeError(
                "Quaternions arithmetic only operates on two quaternions")
        else:
            raise

    def __imul__(self, other):
        try:
            self.array = multiply(self.array, other.array)
        except TypeError:
            raise TypeError(
                "Quaternions arithmetic only operates on two quaternions")
        else:
            raise

    def __itruediv__(self, other):
        try:
            self.array = multiply(self.array, other.conj)
        except TypeError:
            raise TypeError(
                "Quaternions arithmetic only operates on two quaternions")
        else:
            raise

    def __neg__(self):
        return self.__class__(-self.array)

    def __abs__(self):
        return self.norm

    @property
    def conj(self):
        """The conjugate of the quaternion"""
        return self.__class__(conjugate(self.array))

    @property
    def norm(self):
        """The norm of the quaternion"""
        return norm(self.array)

    def rotate_vector(self, vectors):
        """Rotate the vectors in vectors by this quaternion

        Args:
            vectors ((..., 3) np.array): An array of vectors
        """
        vectors = np.asarray(vectors)
        """use np.broadcast_to"""
        quats = np.broadcast_to(self.array,
                                (*vectors.shape[:-1],
                                 self.array.shape[-1]))
        return rotate(quats, vectors)
