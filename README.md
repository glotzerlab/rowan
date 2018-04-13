# README

[![ReadTheDocs Status](https://readthedocs.org/projects/rowan/badge/?version=latest)](http://rowan.readthedocs.io/en/latest/?badge=latest)
[![CircleCI](https://circleci.com/bb/glotzer/rowan.svg?style=svg)](https://circleci.com/bb/glotzer/rowan)
[![Codecov](https://codecov.io/bb/glotzer/rowan/branch/master/graph/badge.svg)](https://codecov.io/bb/glotzer/rowan)
[![PyPI](https://img.shields.io/pypi/v/rowan.svg)](https://pypi.org/project/rowan/)


The rowan package provides a simple and consistent interface for using quaternions.
The package is built entirely on top of numpy and represents quaternions using numpy arrays, meaning that all functions support arbitrarily high-dimensional arrays of quaternions.
Quaternions are encoded as arrays of shape `(...,4)`, with the convention that the final dimension of an array `(a, b, c, d)` represents the quaternion `a + bi + cj + dk`.

## Authors

* Vyas Ramasubramani, vramasub@umich.edu (Maintainer)

## Setup

### Installation from source

To install from source, execute:
```bash
git clone https://bitbucket.org/vramasub/rowan.git
cd rowan
python setup.py install --user
```

### Requirements

* Python = 2.7, >= 3.4
* Numpy >= 1.10

## Testing

The package is currently tested for python versions 2.7, 3.4, 3.5, and 3.6 on Unix.
Continuous integrated testing is performed using CircleCI on these python versions with numpy versions 1.10 and above.

To run the packaged unit tests, execute:

```bash
python -m unittest discover tests
```

To check test coverage, make sure the coverage module is installed:

```bash
pip install coverage
```
    
and then run the packaged unit tests:

```bash
coverage run -m unittest discover tests
```

## Quickstart
This library can be used to work with quaternions by simply instantiating the appropriate numpy arrays and passing them to the required functions.
For example:

```python
>>> import rowan
>>> one = np.array([10, 0, 0, 0])
>>> one_unit = rowan.normalize(one)
>>> assert(one_unit == np.array([1, 0, 0, 0]))
>>> if not one_unit == rowan.quat_multiply(one_unit, one_unit):
>>>     raise RuntimeError("Multiplication failed!")
>>>
>>> one_vec = np.array([1, 0, 0])
>>> rotated_vector = rowan.rotate(one_unit, one_vec)
>>>
>>> import numpy as np
>>> mat = np.eye(3)
>>> quat_rotate = rowan.from_matrix(mat)
>>> alpha, beta, gamma = rowan.to_euler(quat_rotate)
>>> quat_rotate_returned = rowan.from_euler(alpha, beta, gamma)
>>> identity = rowan.to_matrix(quat_rotate_returned)
```

## Documentation
Documentation for rowan is written in [reStructuredText](http://docutils.sourceforge.net/rst.html) and compiled using [Sphinx](http://www.sphinx-doc.org/en/master/).
To build the documentation, first install Sphinx:

```bash
pip install sphinx
```

You can then use sphinx to create the actual documentation in either pdf or HTML form by running the following commands in the rowan root directory:

```bash
cd doc
make html # For html output
make latexpdf # For a LaTeX compiled PDF file
open build/html/index.html
```
