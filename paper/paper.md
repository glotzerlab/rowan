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
date: 20 April 2018
bibliography: paper.bib
---

# Summary

From particle simulations to attitude determination, numerous tasks in science,
engineering, and computer graphics require clear and efficient methods of
defining spatial rotations. Of the many different formalisms used for this
purpose, quaternions are one of the most popular because of their natural
parameterization of rotation space and the efficiency with which
quaternion-based rotations can be implemented. An efficient implementation of
quaternion operations is therefore critical to developing performant
code to solve these more domain-specific problems. Python implementations of
quaternion operations do exist, but they suffer from poor performance [pyquat]
or complex dependencies and installation protocols [numpy-quat] that make them
unsuitable for direct incorporation into other code suites.

The *rowan* package, named for William Rowan Hamilton, is designed to address
these issues. By operating directly on NumPy arrays, *rowan* enables efficient
operations on large arrays of data with the same functions used to operate on
single quaternions. The package is written in pure Python and its only hard
dependency is NumPy, making it an unobtrusive dependency to introduce into other
code bases. NumPy broadcasting has first-class support throughout the package,
making it highly efficient for operating on the large number of arrays common in
graphics or computer simulation applications. Although rotations are only
represented by unit quaternions, *rowan* works equally well for all quaternions
irrespective of the norm. In addition to providing core quaternion operations,
*rowan* can also interconvert between various common rotation formalisms,
perform quaternion interpolation and calculus, generate random rotation
quaternions, compute distances on the quaternion manifold, and perform basic
point set registration.

This package arose due to the proliferation of duplicated and fragmented
quaternion code in numerous different packages developed by the Glotzer Group at
the University of Michigan. Much of this code was also written to work with
individual quaternions rather than arrays, making it highly inefficient for
large sets of operations since it required looping in Python rather than using
the appropriately broadcasted NumPy operation.  By providing a centralized and
highly efficient solution, this package will result in a faster, more modular,
and more easily maintained code base. In addition, *rowan* will be a major aid
to individuals writing code for their individual research, a task that
frequently involves rotating large data sets to generate specific configurations
or compare different systems.


Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

Figures can be included like this: ![Example figure.](figure.png)

# Acknowledgements

We would like to acknowledge Carl S. Adorf, Matthew P. Spellings, and Bradley
D. Dice for helpful suggestions and discussions during the development of this
package.

# References
