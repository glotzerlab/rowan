from setuptools import setup, find_packages

# For the long description, if desired can read directly from README.md
def get_readme():
    with open('README.md') as f:
        return f.read()

setup(name='quaternion',
      version='0.1',
      description='Perform quaternion operations using numpy arrays',
      long_description = get_readme(),
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
        'Programming Language :: Python :: 3.4',
        'Topic :: Scientific/Engineering :: Mathematics',
      ],
      )
