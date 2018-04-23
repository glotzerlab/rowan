.. _development:

=================
Development Guide
=================


Philosophy
==========

The goal of rowan is to provide a flexible, easy-to-use, and scalable approach to dealing with rotation representations.
To ensure maximum flexibility, rowan operates entirely on NumPy arrays, which serve as the *de facto* standard for efficient multi-dimensional arrays in Python.
To be available for a wide variety of applications, rowan aims to work for arbitrarily shaped NumPy arrays, mimicking `NumPy broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_ to the extent possible.
Functions for which this broadcasting is not available should be documented as such.

Since rowan is designed to work everywhere, all hard dependencies aside from NumPy are avoided, although soft dependencies for specific functions are allowed.
To avoid any dependencies on compilers or other software, all rowan code is written in **pure Python**.
This means that while rowan is intended to provide good performance, it may not be the correct choice in cases where performance is critical.
The package was written principally for use-cases where quaternion operations are not the primary bottleneck, so it prioritizes portability, maintainability, and flexibility over optimization.


PEP 20
------
In general, all code in rowan should follow the principles in `PEP 20 <https://www.python.org/dev/peps/pep-0020/>`_.
In particular, prefer simple, explicit code where possible, avoiding unnecessary convolution or complicated code that could be written more simply.
Avoid writing code that is not easy to parse up front.

Inline comments are **highly encouraged**; however, code should be written in a way that it could be understood without comments.
Comments such as "Set x to 10" are not helpful and simply clutter code.
The most useful comments in a package such as rowan are the ones that explain the underlying algorithm rather than the implementations, which should be simple.
For example, the comment "compute the spectral decomposition of A" is uninformative, since the code itself should make this obvious, *e.g*, ``np.linalg.eigh``.
On the other hand, the comment "the eigenvector corresponding to the largest eigenvalue of the A matrix is the quaternion" is instructive.



Source Code Conventions
=======================

All code in rowan should follow `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_ guidelines, which are the *de facto* standard for Python code.
In addition, follow the `Google Python Style Guide <https://google.github.io/styleguide/pyguide.html>`_, which is largely a superset of PEP 8.
Note that Google has amended their standards to match PEP 8's 4 spaces guideline, so write code accordingly.
In particular, write docstrings in the Google style.

Python example:

.. code-block:: python

    # This is the correct style
    def multiply(x, y):
        """Multiply two numbers

        Args:
            x (float): The first number
            y (float): The second number

        Returns:
            The product
        """

    # This is the incorrect style
    def multiply(x, y):
        """Multiply two numbers

        :param x: The first number
        :type x: float
        :param y: The second number
        :type y: float
        :returns: The product
        :rtype: float
        """

Documentation must be included for all files, and is then generated from the docstrings using `sphinx <http://www.sphinx-doc.org/en/stable/index.html>`_.


Unit Tests
==========

All code should include a set of unit tests which test for correct behavior.
All tests should be placed in the ``tests`` folder at the root of the project.
These tests should be as simple as possible, testing a single function each, and they should be kept as short as possible.
Tests should also be entirely deterministic: if you are using a random set of objects for testing, they should either be generated once and then stored in the ``tests/files`` folder, or the random number generator in use should be seeded explicitly (*e.g*, ``numpy.random.seed`` or ``random.seed``).
Tests should be written in the style of the standard Python `unittest <https://docs.python.org/3/library/unittest.html>`_ framework.
At all times, tests should be executable by simply running ``python -m unittest discover tests`` from the root of the project.


General Notes
=============

 * For consistency, NumPy should **always** be imported as ``np`` in code: ``import numpy as np``.
 * Avoid external dependencies where possible, and avoid introducing **any** hard dependencies. Dependencies other than NumPy should always be soft, enabling the rest of the package to function as-is.

Release Guide
=============

To make a new release of rowan, follow the following steps:

#. Make a new branch off of develop based on the expected new version, *e.g.*
   release-2.3.1.
#. Ensure all tests are passing as expected on the new branch. Make any final
   changes as desired on this branch.
#. Once the branch is completely finalized, run bumpversion with the appropriate
   type (patch, minor, major) so that the version now matches the version number
   in the branch name.
#. Once all tests pass on the release branch, merge the branch back into
   develop.
#. Merge develop into master.
#. Generate new source and binary distributions as described in the Python guide
   for `Packaging and distributing projects
   <https://packaging.python.org/tutorials/distributing-packages/#packaging-your-project>`_.
#. Update the conda recipe.
