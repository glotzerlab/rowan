=====
rowan
=====

.. contents::
    :local:

|ReadTheDocs|
|CircleCI|
|Codecov|
|PyPI|
|Codacy|
|Zenodo|
|JOSS|

.. |ReadTheDocs| image:: https://readthedocs.org/projects/rowan/badge/?version=latest
    :target: http://rowan.readthedocs.io/en/latest/?badge=latest
.. |CircleCI| image:: https://circleci.com/gh/glotzerlab/rowan.svg?style=svg
    :target: https://circleci.com/gh/glotzerlab/rowan
.. |Codecov| image:: https://codecov.io/gh/glotzerlab/rowan/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/glotzerlab/rowan
.. |PyPI| image:: https://img.shields.io/pypi/v/rowan.svg
    :target: https://pypi.org/project/rowan/
.. |Codacy| image:: https://api.codacy.com/project/badge/Grade/2ff6c23cb9be4f77827428a87e0e9cfc
    :target: https://www.codacy.com/app/vramasub/rowan?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=glotzerlab/rowan&amp;utm_campaign=Badge_Grade
.. |Zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1323676.svg
    :target: https://doi.org/10.5281/zenodo.1323676
.. |JOSS| image:: http://joss.theoj.org/papers/10.21105/joss.00787/status.svg
    :target: https://doi.org/10.21105/joss.00787)

Welcome to the documentation for rowan, a package for working with quaternions!
Quaternions, which form a number system with various interesting properties, were originally developed for classical mechanics.
Although they have since been largely displaced from this application by vector mathematics, they have become a standard method of representing rotations in three dimensions.
Quaternions are now commonly used for this purpose in various fields, including computer graphics and attitude control.

This package provides tools for standard algebraic operations on quaternions as well as a number of additional tools for *e.g.* measuring distances between quaternions, interpolating between them, and performing basic point-cloud mapping.
A particular focus of the rowan package is working with unit quaternions, which are a popular means of representing rotations in 3D.
In order to provide a unified framework for working with the various rotation formalisms in 3D, rowan allows easy interconversion between these formalisms.

Core features of rowan include (but are not limited to):

* Algebra (multiplication, exponentiation, etc).
* Derivatives and integrals of quaternions.
* Rotation and reflection operations, with conversions to and from matrices, axis angles, etc.
* Various distance metrics for quaternions.
* Basic point set registration, including solutions of the Procrustes problem
  and the Iterative Closest Point algorithm.
* Quaternion interpolation (slerp, squad).

.. toctree::
    :maxdepth: 2
    :caption: Modules:

    package-rowan
    package-calculus
    package-geometry
    package-interpolate
    package-mapping
    package-random
    development

.. toctree::
    :maxdepth: 1
    :caption: Reference:

    license
    changelog
    credits

Getting Started
===============

Requirements
------------

The minimum requirements for using rowan are:

* Python = 2.7, >= 3.3
* NumPy >= 1.10

Installation
------------

The recommended methods for installing rowan are using **pip** or **conda**.
To install the package from PyPI, execute:

.. code-block:: bash

    $ pip install rowan --user

To install the package from conda, first add the **conda-forge** channel and
then install rowan:

.. code-block:: bash

    $ conda config --add channels conda-forge
    $ conda install rowan


If you wish, you may also install rowan by cloning `the repository <https://github.com/glotzerlab/rowan>`_ and running the setup script:

.. code-block:: bash

    $ git clone https://github.com/glotzerlab/rowan.git
    $ cd rowan
    $ python setup.py install --user

Quickstart
----------

This library can be used to work with quaternions by simply instantiating the appropriate NumPy arrays and passing them to the required functions.
For example:

.. code-block:: python

    import rowan
    import numpy as np
    one = np.array([10, 0, 0, 0])
    one_unit = rowan.normalize(one)
    assert(np.all(one_unit == np.array([1, 0, 0, 0])))
    if not np.all(one_unit == rowan.multiply(one_unit, one_unit)):
        raise RuntimeError("Multiplication failed!")

    one_vec = np.array([1, 0, 0])
    rotated_vector = rowan.rotate(one_unit, one_vec)

    mat = np.eye(3)
    quat_rotate = rowan.from_matrix(mat)
    alpha, beta, gamma = rowan.to_euler(quat_rotate)
    quat_rotate_returned = rowan.from_euler(alpha, beta, gamma)
    identity = rowan.to_matrix(quat_rotate_returned)

Running Tests
-------------

The package is currently tested for Python versions 2.7 and Python >= 3.3 on Unix-like systems.
Continuous integrated testing is performed using CircleCI on these Python versions with NumPy versions 1.10 and above.

To run the packaged unit tests, execute the following line from the root of the repository:

.. code-block:: bash

    python -m unittest discover tests

To check test coverage, make sure the coverage module is installed:

.. code-block:: bash

    pip install coverage

and then run the packaged unit tests with the coverage module:

.. code-block:: bash

    coverage run -m unittest discover tests

Running Benchmarks
------------------
Benchmarks for the package are contained in a Jupyter notebook in the `benchmarks` folder in the root of the repository.
If you do not have or do not wish to use the notebook format, an equivalent Benchmarks.py script is also included.
The benchmarks compare rowan to two alternative packages, so you will need to install ``pyquaternion`` and ``numpy_quaternion`` if you wish to see those comparisons.

Building Documentation
----------------------

You can also build this documentation from source if you clone the repository.
The documentation is written in `reStructuredText <http://docutils.sourceforge.net/rst.html>`_ and compiled using `Sphinx <http://www.sphinx-doc.org/en/master/>`_.
To build from source, first install Sphinx:

.. code-block:: bash

    pip install sphinx sphinx_rtd_theme

You can then use Sphinx to create the actual documentation in either PDF or HTML form by running the following commands in the rowan root directory:

.. code-block:: bash

    cd doc
    make html # For html output
    make latexpdf # For a LaTeX compiled PDF file
    open build/html/index.html

Support and Contribution
========================

This package is hosted on `GitHub <https://github.com/glotzerlab/rowan>`_.
Please report any bugs or problems that you find on the `issue tracker <https://github.com/glotzerlab/rowan/issues>`_.

All contributions to rowan are welcomed via pull requests!
Please see the :doc:`development guide <development>` for more information on requirements for new code.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
