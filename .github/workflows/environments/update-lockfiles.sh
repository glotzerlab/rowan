#!/bin/bash

# Execute this script to update all lock files to the latest versions of dependencies.

rm requirements*.txt

for python_version in 3.9 3.10 3.11 3.12 3.13
do
    uv pip compile --python-version ${python_version} --python-platform linux requirements-test.in > requirements-test-${python_version}.txt
done

uv pip compile --python-version 3.13 --python-platform linux requirements-build.in > requirements-build.txt
uv pip compile --python-version 3.12 --python-platform linux requirements-doc.in > requirements-doc.txt
