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
To best serve these goals, **rowan** operates entirely on NumPy arrays (the *de facto* standard for efficient multi-dimensional arrays in Python) and supports `NumPy broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_ wherever possible.
Use of broadcasting ensures that **rowan** can take full advantage of NumPy performance, and in general all operations are very fast (benchmarks are included in the code base).
Furthermore, to remain lightweight and easy to install, **rowan** is written in **pure Python** and has no hard dependencies aside from NumPy.

Code contributions should keep these ideals in mind and adhere to the following guidelines:

  * Use the OneFlow_ model of development:
    - Both new features and bug fixes should be developed in branches based on ``master``.
    - Hotfixes (critical bugs that need to be released *fast*) should be developed in a branch based on the latest tagged release.
  * All code must be compatible with all supported versions of Python (listed in the package ``setup.py`` file).
  * Avoid external dependencies where possible, and avoid introducing **any** hard dependencies. Soft dependencies are allowed for specific functionality, but such dependencies cannot impede the installation of **rowan** or the use of any other features.
  * All code should adhere to the source code and documentation conventions discussed below.
  * Create `unit tests <https://en.wikipedia.org/wiki/Unit_testing>`_  and `integration tests <https://en.wikipedia.org/wiki/Integration_testing>`_ as part of development.
  * Preserve backwards-compatibility whenever possible. Make clear if something must change, and notify package maintainers that merging such changes will require a major release.
  * Enable broadcasting if at all possible. Functions for which broadcasting is not available must be documented as such.


.. _github: https://github.com/glotzerlab/rowan
.. _OneFlow: https://www.endoflineblog.com/oneflow-a-git-branching-model-and-workflow

.. tip::

    During continuous integration, the code is checked automatically with `pre-commit`_.
    To run these checks locally, you can install and run pre-commit like so:

    .. code-block:: console

        python -m pip install pre-commit
        pre-commit run --all-files

    To avoid having commits fail in case you forget to run this, you can set up a git pre-commit hook using `pre-commit`_:

    .. code-block:: console

        pre-commit install

.. _Flake8: http://flake8.pycqa.org/en/latest/
.. _pre-commit: https://pre-commit.com/

.. note::

    Please see the individual package documentation for detailed guidelines on how to contribute to a specific package.


Source Code Conventions
-----------------------

The **rowan** package adheres to a relatively strict set of style guidelines.
All code in **rowan** should be formatted using `ruff`_; a notable consequence of this is that the recommended max line length is 88, not the more common 80.
Imports should be formatted using `isort`_.
For guidance on the style, see `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_ and the `Google Python Style Guide <https://google.github.io/styleguide/pyguide.html>`_, but any ambiguities should be resolved automatically by running black.
All code should also follow the principles in `PEP 20 <https://www.python.org/dev/peps/pep-0020/>`_.
In particular, always prefer simple, explicit code where possible, avoiding unnecessary convolution or complicated code that could be written more simply.
Avoid writing code in a manner that will be difficult for others to understand.

.. _ruff: https://docs.astral.sh/ruff/
.. _isort: https://pycqa.github.io/isort/

Documentation
-------------

API documentation should be written as part of the docstrings of the package in the `Google style <https://google.github.io/styleguide/pyguide.html#383-functions-and-methods>`__.
There is one notable exception to the guide: class properties should be documented in the getters functions, not as class attributes, to allow for more useful help messages and inheritance of docstrings.
Docstrings may be validated using `pydocstyle <http://www.pydocstyle.org/>`__ (or using the flake8-docstrings plugin as documented above).
The `official documentation <https://rowan.readthedocs.io/>`_ is generated from the docstrings using `Sphinx <http://www.sphinx-doc.org/en/stable/index.html>`_.


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
