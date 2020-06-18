import doctest
import pkgutil
import unittest

import rowan


def load_tests(loader, tests, ignore):
    modules = pkgutil.walk_packages(rowan.__path__, rowan.__name__ + '.')
    for _, module_name, _ in modules:
        tests.addTests(doctest.DocTestSuite(module_name, globs={'rowan': rowan}))
    return tests


if __name__ == '__main__':
    unittest.main()
