from setuptools import setup, find_packages

import os

# Gets the version
version = '1.1.0'

# Read README for PyPI, fallback if it fails.
desc = 'Perform quaternion operations using NumPy arrays'
try:
    readme_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'README.md')
    with open(readme_file) as f:
        readme = f.read()
except ImportError:
    readme = desc

setup(name='rowan',
      version=version,
      description=desc,
      long_description=readme,
      long_description_content_type='text/markdown',
      url='http://bitbucket.org/glotzer/rowan',
      author='Vyas Ramasubramani',
      author_email='vramasub@umich.edu',
      packages=find_packages(exclude=["tests"]),
      zip_safe=True,
      install_requires=[
          'numpy>=1.10'
      ],
      python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, <4',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Topic :: Scientific/Engineering :: Mathematics',
      ],
      )
