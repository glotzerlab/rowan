The format is based on `Keep a Changelog <http://keepachangelog.com/en/1.0.0/>`_.
This project adheres to `Semantic Versioning <http://semver.org/spec/v2.0.0.html>`_.


Unreleased
----------

v1.1.6 - 2018-10-18
-------------------

Fixed
+++++

* Fifth try of releasing using CircleCI.

v1.1.5 - 2018-10-18
-------------------

Fixed
+++++

* Fourth try of releasing using CircleCI.

v1.1.4 - 2018-10-18
-------------------

Fixed
+++++

* Third try of releasing using CircleCI.

v1.1.3 - 2018-10-18
-------------------

Fixed
+++++

* Second try of releasing using CircleCI.

v1.1.2 - 2018-10-18
-------------------

Fixed
+++++

* Fix usage of release tag in CircleCI config.

v1.1.1 - 2018-10-18
-------------------

Added
+++++

* Automated deployment using CircleCI.
* Added PDF of paper to the repository.

Fixed
+++++

* Added missing factor of 2 in angle calculation.
* Fixed issue where method was not respected in rowan.mapping.
* Disabled equivalent quaternion feature and test of rowan.mapping, which has a known bug.
* Added missing negative in failing unit test.

v1.1.0 - 2018-07-30
-------------------

Added
+++++

* Included benchmarks including comparison to alternatives.
* Installation instructions in the Sphinx documentation.
* More examples for rowan.mapping.

Changed
+++++++

* All examples in docstrings now use the full paths of subpackages.
* All examples in docstrings import all needed packages aside from rowan.

Fixed
+++++

* Instability in vector\_vector\_rotation for antiparallel vectors.
* Various code style issues.
* Broken example in the Sphinx documentation.

v1.0.0 - 2018-05-29
-------------------

Fixed
+++++

* Numerous style fixes.
* Fix version numbering in the Changelog.

v0.6.1 - 2018-04-20
-------------------

Fixed
+++++

* Use of bumpversion and consistent versioning across the package.

v0.6.0 - 2018-04-20
-------------------

Added
+++++

* Derivatives and integrals of quaternions.
* Point set registration methods and Procrustes analysis.

v0.5.1 - 2018-04-13
-------------------

Fixed
+++++

* README rendering on PyPI.

v0.5.0 - 2018-04-12
-------------------

Added
+++++

* Various distance metrics on quaternion space.
* Quaternion interpolation.

Fixed
+++++

* Update empty __all__ variable in geometry to export functions.


v0.4.4 - 2018-04-10
-------------------

Added
+++++

* Rewrote internals for upload to PyPI.

v0.4.3 - 2018-04-10
-------------------

Fixed
+++++

* Typos in documentation.

v0.4.2 - 2018-04-09
-------------------

Added
+++++

* Support for Read The Docs and Codecov.
* Simplify CircleCI testing suite.
* Minor changes to README.
* Properly update this document.

v0.4.1 - 2018-04-08
-------------------

Fixed
+++++

* Exponential for bases other than e are calculated correctly.

v0.4.0 - 2018-04-08
-------------------

Added
+++++

* Add functions relating to exponentiation: exp, expb, exp10, log, logb, log10, power.
* Add core comparison functions for equality, closeness, finiteness.

v0.3.0 - 2018-03-31
-------------------

Added
+++++

* Broadcasting works for all methods.
* Quaternion reflections.
* Random quaternion generation.

Changed
+++++++

* Converting from Euler now takes alpha, beta, and gamma as separate args.
* Ensure more complete coverage.

v0.2.0 - 2018-03-08
-------------------

Added
+++++

* Added documentation.
* Add tox support.
* Add support for range of python and numpy versions.
* Add coverage support.

Changed
+++++++

* Clean up CI.
* Ensure pep8 compliance.

v0.1.0 - 2018-02-26
-------------------

Added
+++++
* Initial implementation of all functions.
