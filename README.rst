README
======

|ReadTheDocs Status| |CircleCI| |Codecov|

The hamilton package provides a simple and consistent interface for
using quaternions. The package is built entirely on top of numpy and
represents quaternions using numpy arrays, meaning that all functions
support arbitrarily high-dimensional arrays of quaternions. Quaternions
are encoded as arrays of shape ``(...,4)``, with the convention that the
final dimension of an array ``(a, b, c, d)`` represents the quaternion
``a + bi + cj + dk``.

Authors
-------

-  Vyas Ramasubramani, vramasub@umich.edu (Maintainer)

Setup
-----

Installation from source
~~~~~~~~~~~~~~~~~~~~~~~~

To install from source, execute:

::

    git clone https://bitbucket.org/vramasub/hamilton.git
    cd hamilton
    python setup.py install --user

Requirements
~~~~~~~~~~~~

-  Python = 2.7, >= 3.4
-  Numpy >= 1.10

Testing
-------

The package is currently tested for python versions 2.7, 3.4, 3.5, and
3.6 on Unix. Continuous integrated testing is performed using CircleCI
on these python versions with numpy versions 1.10 and above.

To run the packaged unit tests, execute:

::

    python -m unittest discover tests

To check test coverage, make sure the coverage module is installed:

::

    pip install coverage

and then run the packaged unit tests:

::

    coverage run -m unittest discover tests

Quickstart
----------

This library can be used to work with quaternions by simply
instantiating the appropriate numpy arrays and passing them to the
required functions. For example:

::

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

Documentation
-------------

Documentation for hamilton is written in
`reStructuredText <http://docutils.sourceforge.net/rst.html>`__ and
compiled using `Sphinx <http://www.sphinx-doc.org/en/master/>`__. To
build the documentation, first install Sphinx:

::

    pip install sphinx

You can then use sphinx to create the actual documentation in either pdf
or HTML form by running the following commands in the hamilton root
directory:

::

    cd doc
    make html # For html output
    make latexpdf # For a LaTeX compiled PDF file
    open build/html/index.html

.. |ReadTheDocs Status| image:: https://readthedocs.org/projects/hamilton/badge/?version=latest
   :target: http://hamilton.readthedocs.io/en/latest/?badge=latest
.. |CircleCI| image:: https://circleci.com/bb/glotzer/hamilton.svg?style=svg
   :target: https://circleci.com/bb/glotzer/hamilton
.. |Codecov| image:: https://codecov.io/bb/glotzer/hamilton/branch/master/graph/badge.svg
   :target: https://codecov.io/bb/glotzer/hamilton
