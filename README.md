# README

[![ReadTheDocs Status](https://readthedocs.org/projects/rowan/badge/?version=latest)](http://rowan.readthedocs.io/en/latest/?badge=latest)
[![CircleCI](https://circleci.com/bb/glotzer/rowan.svg?style=svg)](https://circleci.com/bb/glotzer/rowan)
[![Codecov](https://codecov.io/bb/glotzer/rowan/branch/master/graph/badge.svg)](https://codecov.io/bb/glotzer/rowan)
[![PyPI](https://img.shields.io/pypi/v/rowan.svg)](https://pypi.org/project/rowan/)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/44a7677f2d7341e68a8338d1513f71e9)](https://www.codacy.com/app/vramasub/rowan)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/rowan.svg)](https://bitbucket.org/glotzer/rowan/)


Welcome to rowan, a python package for quaternions.
The package is built entirely on top of NumPy and represents quaternions using NumPy arrays, meaning that all functions support arbitrarily high-dimensional arrays of quaternions.
Quaternions are encoded as arrays of shape `(...,4)`, with the convention that the final dimension of an array `(a, b, c, d)` represents the quaternion `a + bi + cj + dk`.
The package covers all basic quaternion algebraic and calculus operations, and also provides features for measuring distances, performing point cloud mapping, and interpolating.
If you have any questions about how to work with rowan, please visit the
[ReadTheDocs page](http://rowan.readthedocs.io/en/latest/).

## Authors

* Vyas Ramasubramani <vramasub@umich.edu> (Lead developer)

## Setup

The recommended methods for installing rowan are using **pip** or **conda**.

### Installation via pip

To install the package from PyPI, execute:
```bash
pip install rowan --user
```

### Installation via conda

To install the package from conda, first add the **conda-forge** channel:
```bash
conda config --add channels conda-forge
```

After the **conda-forge** channel has been added, you can install rowan by
executing
```bash
conda install rowan
```

### Installation from source

To install from source, execute:
```bash
git clone https://bitbucket.org/glotzer/rowan.git
cd rowan
python setup.py install --user
```

### Requirements

* Python = 2.7, >= 3.3
* NumPy >= 1.10

## Testing

The package is currently tested for Python versions 2.7, 3.3, 3.4, 3.5, 3.6, and 3.7 on Unix-like systems.
Continuous integrated testing is performed using CircleCI on these Python versions with NumPy versions 1.10 and above.

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
This library can be used to work with quaternions by simply instantiating the appropriate NumPy arrays and passing them to the required functions.
For example:

```python
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
```

## Documentation
Documentation for rowan is written in [reStructuredText](http://docutils.sourceforge.net/rst.html) and compiled using [Sphinx](http://www.sphinx-doc.org/en/master/).
To build the documentation, first install Sphinx:

```bash
pip install sphinx
```

You can then use Sphinx to create the actual documentation in either PDF or HTML form by running the following commands in the rowan root directory:

```bash
cd doc
make html # For html output
make latexpdf # For a LaTeX compiled PDF file
open build/html/index.html
```
