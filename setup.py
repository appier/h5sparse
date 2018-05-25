#!/usr/bin/env python
from setuptools import setup


setup_requires = [
    'nose',
    'coverage',
]
install_requires = [
    'h5py',
    'numpy',
    'scipy',
    'six',
]
tests_require = []

description = "Scipy sparse matrix in HDF5."

long_description = """\
Please visit  the `Github repository <https://github.com/appier/h5sparse>`_
for more information.\n
"""
with open('README.rst') as fp:
    long_description += fp.read()


setup(
    name='h5sparse',
    version="0.0.5",
    description=description,
    long_description=long_description,
    author='Appier Inc.',
    url='https://github.com/appier/h5sparse',
    setup_requires=setup_requires,
    install_requires=install_requires,
    tests_require=tests_require,
    license="MIT",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Database',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
    ],
    test_suite='nose.collector',
    packages=[
        'h5sparse',
    ],
)
