#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#MIT License
#
#Copyright (c) 2018 Justin Shenk
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

import sys
from setuptools import setup, setuptools
from fer import __version__

__author__ = 'Justin Shenk'


def readme():
    with open('README.md', encoding="UTF-8") as f:
        return f.read()


if sys.version_info < (3, 6, 0):
    sys.exit('Python < 3.6.0 is not supported!')

setup(
    name='fer',
    version=__version__,
    description='Facial Expression Recognition based on Keras',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/justinshenk/fer',
    author='Justin Shenk',
    author_email='shenk.justin@gmail.com',
    license='MIT',
    packages=setuptools.find_packages(exclude=["tests.*", "tests"]),
    install_requires=[
        'matplotlib', 'tensorflow', 'opencv-contrib-python', 'keras', 'pandas'
    ],
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
    include_package_data=True,
    keywords="expression emotion detection tensorflow pip package",
    entry_point={
        'console_scripts': [
            'fer=fer.fer:inference',
        ],
    },
    zip_safe=False)
