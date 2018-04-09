The format is based on `Keep a Changelog <http://keepachangelog.com/en/1.0.0/>`_
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).


Unreleased
----------

v0.4.1 - 2018-04-08
-----------------

Fixed
+++++

* Exponential for bases other than e are calculated correctly.

v0.4.0 - 2018-04-08
-----------------

Added
+++++

* Add functions relating to exponentiation: exp, expb, exp10, log, logb, log10, power
* Add core comparison functions for equality, closeness, finiteness

v0.3.0 - 2018-03-31
-----------------

Added
+++++

* Broadcasting works for all methods
* Quaternion reflections
* Random quaternion generation

Changed
+++++++

* Converting from Euler now takes alpha, beta, and gamma as separate args
* Ensure more complete coverage

v0.2.0 - 2018-03-08
-----------------

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
-----------------

Added
+++++
* Initial implementation of all functions.
