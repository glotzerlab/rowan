from setuptools import setup, find_packages

import os

# Gets the version
version = '1.2.2'

# Read README for PyPI, fallback if it fails.
desc = 'Perform quaternion operations using NumPy arrays'
try:
    readme_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'README.md')
    with open(readme_file) as f:
        readme = f.read()
except ImportError:
    readme = desc

# Supported versions are determined according to NEP 29.
# https://numpy.org/neps/nep-0029-deprecation_policy.html

setup(name='rowan',
      version=version,
      description=desc,
      long_description=readme,
      long_description_content_type='text/x-rst',
      url='https://github.com/glotzerlab/rowan',
      author='Vyas Ramasubramani',
      author_email='vramasub@umich.edu',
      packages=find_packages(exclude=["tests"]),
      zip_safe=True,
      install_requires=[
          'numpy>=1.15'
      ],
      python_requires='>=3.6, <4',
      classifiers=[
          'Development Status :: 6 - Mature',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Topic :: Scientific/Engineering :: Mathematics',
      ],
      )
