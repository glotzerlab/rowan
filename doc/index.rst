rowan
=====

Welcome to the documentation for rowan, a package for working with quaternions!
Quaternions form a number system with various interesting properties, and they have a number of uses.
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

To install rowan, you have a few options.
The package can either be installed through PyPI:

.. code-block:: bash
    $ pip install rowan --user

using conda

.. code-block:: bash
    $ conda install rowan

or by cloning the repository `from source <https://bitbucket.org/glotzer/rowan>`_
and running setuptools

.. code-block:: bash
    $ git clone https://bitbucket.org/glotzer/rowan.git
    $ python setup.py install --user

Note that the conda installation requires that you first add the **conda-forge**
channel.

.. toctree::
    :maxdepth: 2
    :caption: Contents:

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

Support and Contribution
========================

This package is hosted on `Bitbucket <https://bitbucket.org/glotzer/rowan>`_.
Please report any bugs or problems that you find on the `issue tracker <https://bitbucket.org/glotzer/rowan/issues>`_.

All contributions to rowan are welcomed via pull requests!
Please see the :doc:`development guide <development>` for more information on requirements for new code.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
