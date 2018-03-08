.. hamilton documentation master file, created by sphinx-quickstart on Mon Feb 26 21:45:57 2018.

hamilton
========

Welcome to the documentation for hamilton!
The hamilton package addresses the need for a unified framework for working with the various rotation formalisms in 3D.
According to `Euler's rotation theorem <https://en.wikipedia.org/wiki/Euler%27s_rotation_theorem>`_, all rotations and orientations can be represented by three numbers.
In practice, however, there are numerous ways to represent these three degrees of freedom, including rotation matrices, Euler angles, Euler axis-angles, and quaternions.
Named for William Rowan Hamilton, who invented quaternions and popularized their use, hamilton focuses on quaternions, but it also provides utilities for interconverting between quaternions and the other common rotation representations as well.

To install hamilton, first clone the repository `from source <https://bitbucket.org/glotzer/hamilton>`_.
Once installed, the package can be installed using setuptools::

    $ python setup.py install --user

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    package-hamilton
    development

.. toctree::
    :maxdepth: 1
    :caption: Reference:

    license
    changelog
    credits

Support and Contribution
========================

This package is hosted on `Bitbucket <https://bitbucket.org/glotzer/hamilton>`_.
Please report any bugs or problems that you find on the `issue tracker <https://bitbucket.org/glotzer/hamilton/issues>`_.

All contributions to hamilton are welcomed!
Please see the :doc:`development guide <development>` for more information.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
