import unittest
import doctest
import rowan
from rowan import functions
import inspect


def load_tests(loader, tests, ignore):
    optionflags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    tests.addTests(doctest.DocTestSuite(
        functions, optionflags=optionflags, globs={'rowan': rowan}))
    #  for name, member in inspect.getmembers(rowan):
        #  if inspect.ismodule(member):
            #  tests.addTests(doctest.DocTestSuite(
                #  member, optionflags=optionflags))
    return tests


if __name__ == '__main__':
    unittest.main()
