"""Test examples in code docstrings using doctest."""

import doctest
import pkgutil
import unittest

import rowan


def load_tests(loader, tests, ignore):
    """Create tests from all docstrings by walking the package hierarchy."""
    modules = pkgutil.walk_packages(rowan.__path__, rowan.__name__ + ".")
    for _, module_name, _ in modules:
        tests.addTests(doctest.DocTestSuite(module_name, globs={"rowan": rowan}))
    return tests


if __name__ == "__main__":
    unittest.main()
