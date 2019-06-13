.. _development:

=================
Development Guide
=================


All contributions to **rowan** are welcome!
Developers are invited to contribute to the framework by pull request to the package repository on `github`_, and all users are welcome to provide contributions in the form of **user feedback** and **bug reports**.
We recommend discussing new features in form of a proposal on the issue tracker for the appropriate project prior to development.


Design Philosophy and Code Guidelines
=====================================

The goal of **rowan** is to provide a flexible, easy-to-use, and scalable approach to dealing with rotation representations.
To ensure maximum flexibility, **rowan** operates entirely on NumPy arrays, which serve as the *de facto* standard for efficient multi-dimensional arrays in Python.
To be available for a wide variety of applications, **rowan** works for arbitrarily shaped NumPy arrays, mimicking `NumPy broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_ to the maximum extent possible.
**rowan** is meant to be as lightweight and easy to install as possible.
Although it is designed to provide good performance, it is written in **pure Python** and as such may not be the correct choice in cases where the performance of quaternion operations is a critical bottleneck.

All code contributed to **rowan** must adhere to the following guidelines:

  * Use the OneFlow_ model of development:
    - Both new features and bug fixes should be developed in branches based on ``master``.
    - Hotfixes (critical bugs that need to be released *fast*) should be developed in a branch based on the latest tagged release.
  * All code must be compatible with all supported versions of Python (listed in the package ``setup.py`` file).
  * Avoid external dependencies where possible, and avoid introducing **any** hard dependencies. Soft dependencies are allowed for specific functionality, but such dependencies cannot impede the installation of **rowan** or the use of any other features.
  * All code should adhere to the source code conventions discussed below.
  * Follow the rules for documentation discussed below.
  * Create `unit tests <https://en.wikipedia.org/wiki/Unit_testing>`_  and `integration tests <https://en.wikipedia.org/wiki/Integration_testing>`_ that cover the common cases and the corner cases of the code (more information below).
  * Preserve backwards-compatibility whenever possible. Make clear if something must change, and notify package maintainers that merging such changes will require a major release.
  * Enable broadcasting if at all possible. Functions for which broadcasting is not available must be documented as such.
  * For consistency, NumPy should **always** be imported as ``np`` in code: ``import numpy as np``.

.. _github: https://github.com/glotzerlab/rowan
.. _OneFlow: https://www.endoflineblog.com/oneflow-a-git-branching-model-and-workflow

.. tip::

    During continuous integration, the code is checked automatically with `Flake8`_.
    Run the following commands to set up a pre-commit hook that will ensure your code is compliant before committing:

    .. code-block:: bash

        flake8 --install-hook git
        git config --bool flake8.strict true


.. _Flake8: http://flake8.pycqa.org/en/latest/

.. note::

    Please see the individual package documentation for detailed guidelines on how to contribute to a specific package.


Source Code Conventions
-----------------------

All code in rowan should follow `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_ guidelines, which are the *de facto* standard for Python code.
In addition, follow the `Google Python Style Guide <https://google.github.io/styleguide/pyguide.html>`_, which is largely a superset of PEP 8.
Note that Google has amended their standards to match PEP 8's 4 spaces guideline, so write code accordingly.

All code should follow the principles in `PEP 20 <https://www.python.org/dev/peps/pep-0020/>`_.
In particular, always prefer simple, explicit code where possible, avoiding unnecessary convolution or complicated code that could be written more simply.
Avoid writing code in a manner that will be difficult for others to understand.


Documentation
-------------

API documentation should be written as part of the docstrings of the package.
All docstrings should be written in the Google style.

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

Documentation must be included for all functions in all files.
The `official documentation <https://rowan.readthedocs.io/>`_ is generated from the docstrings using `Sphinx <http://www.sphinx-doc.org/en/stable/index.html>`_.

In addition to API documentation, inline comments are **highly encouraged**.
Code should be written as transparently as possible, so the primary goal of documentation should be explaining the algorithms or mathematical concepts underlying the code.
Avoid comments that simply restate the nature of lines of code.
For example, the comment "compute the spectral decomposition of A" is uninformative, since the code itself should make this obvious, *e.g*, ``np.linalg.eigh``.
On the other hand, the comment "the eigenvector corresponding to the largest eigenvalue of the A matrix is the quaternion" is instructive.


Unit Tests
----------

All code should include a set of unit tests which test for correct behavior.
All tests should be placed in the ``tests`` folder at the root of the project.
These tests should be as simple as possible, testing a single function each, and they should be kept as short as possible.
Tests should also be entirely deterministic: if you are using a random set of objects for testing, they should either be generated once and then stored in the ``tests/files`` folder, or the random number generator in use should be seeded explicitly (*e.g*, ``numpy.random.seed`` or ``random.seed``).
Tests should be written in the style of the standard Python `unittest <https://docs.python.org/3/library/unittest.html>`_ framework.
At all times, tests should be executable by simply running ``python -m unittest discover tests`` from the root of the project.


Release Guide
=============

To make a new release of rowan, follow the following steps:

#. Make a new branch off of master based on the expected new version, *e.g.*
   release-2.3.1.
#. Make any final changes as desired on this branch. Push the changes and
   ensure all tests are passing as expected on the new branch.
#. Once the branch is completely finalized, run bumpversion with the
   appropriate type (patch, minor, major) so that the version now matches the
   version number in the branch name.
#. Merge the branch back into master, then push master and push tags. The
   tagged commit will automatically trigger generation of binaries and upload
   to PyPI and conda-forge.
#. Delete the release branch both locally and on the remote.
