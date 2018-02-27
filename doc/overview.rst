Overview
========

Rotations and orientations in three dimensions are of paramount importance in various systems.
Various dynamical properties such as moments of inertia depend on the rotational behavior of systems, and accurately representing three-dimensional many-body systems requires knowledge of their individual orientations. 
`Euler's rotation theorem <https://en.wikipedia.org/wiki/Euler%27s_rotation_theorem>`_ tells us that all rotations can be represented by three numbers, a fact that also holds for orientations.
However, a variety of different conventions exist for the representation of these rotations and orientations, including rotation matrices, Euler angles, Euler axis-angles, and quaternions.
Although inconvenient, each of these representations has various strengths and weaknesses that can be leveraged in different situations.

The hamilton package addresses the critical need for a unified framework for working with these representations.
Named for William Rowan Hamilton, who invented quaternions and popularized their use, the hamilton package is centered around quaternions, but it provides utilities for interconverting between quaternions and the other rotation formalisms mentioned above.


Philosophy
----------

The goal of hamilton is to provide a flexible, easy-to-use, and scalable approach to dealing with rotation representations.
To ensure maximum flexibility, hamilton operates entirely on numpy arrays, which serve as the *de facto* standard for efficient multi-dimensional arrays in Python.
To be available for a wide variety of applications, hamilton aims to work for arbitrarily shaped numpy arrays, mimicking `numpy broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_ to the extent possible.
Functions for which this broadcasting is not available should be documented as such.

Since hamilton is designed to work everywhere, all hard dependencies aside from numpy are avoided, although soft dependencies for specific functions are allowed.
To avoid any dependencies on compilers or other software, all hamilton code is written in **pure Python**.
This means that while hamilton is intended to provide good performance, it may not be the correct choice in cases where performance is critical.
The package was written principally for use-cases where quaternion operations are not the primary bottleneck, so it prioritizes maintainability, reusability, and flexibility over optimization.
