from setuptools import setup, find_packages

import os
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'hamilton', '_version.py')) as f:
    exec(f.read())

setup(name='hamilton',
      version=__version__,
      description='Perform quaternion operations using numpy arrays',
      url='http://github.com/vramasub/quaternion',
      author='Vyas Ramasubramani',
      author_email='vramasub@umich.com',
      packages=find_packages(exclude=["tests"]),
      zip_safe=True,

      install_requires=[
          'numpy>=1.10'
      ],

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
