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

Numerous fields in science and engineering require methods for rotating
objects. Of the different formalisms for representing spatial rotations,
quaternions are perhaps the most popular due to their natural parameterization
of the space of rotations $SO(3)$ and the relative efficiency with which
quaternion-based rotation operations can be computed. A simple, uniform, and
efficient implementation of quaternion operations is therefore critical to
developing code to solve domain-specific problems in areas such as particle
simulation and attitude determination. Python implementations of quaternion
operations do exist, but they suffer from various drawbacks. Some tools are
performance limited due to, *e.g.*, having limited or no support for NumPy
style array broadcasting [@pyquat]. Since NumPy is a *de facto* standard in
scientific computing applications, such support is both a prerequisite for a
package to be easily incorporated into existing code bases and a Pythonic way
to achieve a performant solution. Other packages that do support NumPy may have
complex dependencies for accessing their full features or require conversion
into some internal format, making them cumbersome to incorporate into existing
code bases that need to operate on raw arrays [@npquat].

The *rowan* package, named for William Rowan Hamilton, is a quaternion package
that addresses these issues. By operating directly on NumPy arrays and offering
first-class support for broadcasting throughout the package, *rowan* ensures
high efficiency for operating on the large arrays common in computer graphics
or simulation applications. The package avoids any hard dependencies other than
NumPy itself and uses NumPy arrays as a universal language, making *rowan* an
unobtrusive dependency with essentially zero barrier for introduction into
existing code bases. A full-featured quaternion library, *rowan* includes
extensive features in addition to basic quaternion arithmetic operations. These
functions include: methods for point set registration, including some that are
specialized for solving the Procrustes problem of superimposing corresponding
sets of points; functions for quaternion calculus and interpolation; the
ability to sample random rotation quaternions from $SO(3)$; and functions to
compute various distance metrics on the quaternion manifold. For applications
focused on rotations, *rowan* provides the ability to convert between numerous
common rotation formalisms, including full support for all Euler angle
conventions, which is not found in other Python quaternion packages.

This package arose due to the proliferation of fragmented quaternion code in
disparate code bases developed by the Glotzer Group at the University of
Michigan. Each of these code bases requires different sets of features and
levels of generality. *rowan* addresses these needs by providing a unified,
high-performance, easily utilized solution. The package was incorporated into
the open source plato [@plato] simulation visualization tool as well some
internal packages that have not yet been open sourced. Going forward, *rowan*
will not only simplify the maintenance of many of our existing code bases, it
will also simplify code development involving quaternion operations going
forward, both within and outside our group.

# Acknowledgements

This work was partially supported by a Simons Investigator award from the
Simons Foundation to Sharon Glotzer. We would like to acknowledge Carl S.
Adorf, Matthew P. Spellings, and Bradley D. Dice for helpful suggestions and
discussions during the development of this package.

# References
