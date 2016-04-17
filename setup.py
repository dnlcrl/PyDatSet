#!/usr/bin/env python
# coding: utf-8

from distutils.core import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="pydatset",
    author="Daniele Ettore Ciriello",
    author_email="ciriello.daniele@gmail.com",
    version="0.1",
    license="MIT",
    url="https://github.com/dnlcrl/PyDatSet",
    download_url="https://github.com/dnlcrl/PyDatSet",
    description="Load and augment various datasets in Python for computer vision purposes",
    py_modules="",
    packages=['pydatset'],
    install_requires=required,
    scripts=""
)
