from setuptools import setup, find_packages

setup(name='quaternion',
      version='0.1',
      description='Perform quaternion operations using numpy arrays',
      url='http://github.com/vramasub/quaternion',
      author='Vyas Ramasubramani',
      author_email='vramasub@umich.com',
      packages=find_packages(exclude=["tests"]),
      zip_safe=True,

      install_requires=[
          'numpy>=1.13'
          ],

      classifiers=[
        'Development Status :: 3 - Alpha',
        #'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Topic :: Scientific/Engineering :: Mathematics',
      ],
      )
