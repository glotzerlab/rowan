---
title: 'rowan: A Python package for working with quaternions'
tags:
  - Python
  - physics
  - graphics
  - rotation
  - orientation
authors:
  - name: Vyas Ramasubramani
    orcid: 0000-0001-5181-9532
    affiliation: 1
  - name: Sharon C. Glotzer
    orcid: 0000-0002-7197-0085
    affiliation: "1, 2, 3"
affiliations:
 - name: Department of Chemical Engineering, University of Michigan
   index: 1
 - name: Department of Materials Science and Engineering, University of Michigan
   index: 2
 - name: Biointerfaces Institute, University of Michigan
   index: 3
date: 4 May 2018
bibliography: paper.bib
---

# Summary

Numerous fields in science and engineering require methods for working with
spatial rotations. Of the many different formalisms for representing these
rotations, quaternions are perhaps the most popular due to their natural
parameterization of the space of rotations $SO(3)$ and the relative efficiency
with which quaternion-based rotation operations can be computed. A simple,
uniform, and efficient implementation of quaternion operations is therefore
critical to developing code to solve domain-specific problems in areas such as
particle simulation and attitude determination. Python implementations of
quaternion operations do exist, but they suffer from performance drawbacks due
to having limited or no support for broadcasting [@pyquat].
Additionally, some options have complex dependencies for accessing their full
features or require conversion into some internal format, making them cumbersome
to incorporate into existing code bases that need to operate on raw arrays
[@npquat].

The *rowan* package, named for William Rowan Hamilton, is a quaternion package
that addresses these issues. By operating directly on NumPy arrays and offering
first-class support for broadcasting throughout the package, *rowan* ensures
high efficiency for operating on the large arrays common in computer graphics or
simulation applications. The package avoids any hard dependencies other than
NumPy itself, which, in conjunction with its use of NumPy arrays as a consistent
language, makes it an unobtrusive dependency with essentially zero barrier for
introduction into existing code bases with almost no changes. A full-featured
quaternion library, *rowan* includes extensive features in addition to basic
quaternion arithmetic operations. These functions include solutions to the
Procrustes problem and basic point set registration, quaternion calculus and
interpolation, the ability to correctly sample random rotation quaternions from
$SO(3)$, and various distance metrics on the quaternion manifold. For
applications focused on rotations, *rowan* provides the ability to convert
between numerous common rotation formalisms, including full support for
all Euler angle conventions, which is not found in other Python quaternion code
bases.

This package arose due to the proliferation of fragmented quaternion code in
disparate code bases developed by the Glotzer Group at the University of
Michigan. Each of these code bases requires different sets of features and
levels of generality, so *rowan* attempts to address these needs by providing a
unified, efficient solution. The package was incorporated into the open source
Plato [@plato] simulation visualization tool as well some internal packages that
have not yet been open sourced. In addition to improving the maintainability of
these other packages by providing a modular solution for quaternion operations,
*rowan* will aid individuals writing code for their own personal purposes.

# Acknowledgements

We would like to acknowledge Carl S. Adorf, Matthew P. Spellings, and Bradley D.
Dice for helpful suggestions and discussions during the development of this
package.

# References
