#!/usr/bin/env python
#coding: utf-8

from distutils.core import setup

setup(
	name = "pydatset",
	author = "Daniele Ettore Ciriello",
	author_email = "ciriello.daniele@gmail.com",
	version = "0.1",
	license = "MIT",
	url = "https://github.com/dnlcrl/PyDatSet",
	download_url = "https://github.com/dnlcrl/PyDatSet",
	description = "Load and augment various datasets in Python for computer vision purposes",
	py_modules = "",
	packages = ['pydatset'],
	install_requires = ['numpy', 'scipy', 'cv2', 'scikit_image'],
	scripts = ""
)