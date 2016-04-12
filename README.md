# PyDatSet

Load various datasets for image recognition purposes

## Description 

This repo contains pydatset, a package for loading (and eventually augmenting) datasets in python, you can check the source of each function into their pydocs. Currently supported datasets are [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), [Tiny-ImageNet](http://cs231n.stanford.edu/project.html) and [GTSRB](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news). 

Pull requests are welcome!!!

## Requirements

- [numpy](www.numpy.org/)
- [scipy](www.scipy.org/) (used to load Tiny-ImageNet)
- [cv2](opencv.org) (used to load GTSRB)
- [scikit_learn](scikit-learn.org/) (used for data augmentation)

## Installation

- You can get [pip](https://pypi.python.org/pypi/pip) and install everything by running:

		pip install git+git://github.com/dnlcrl/PyDatSet.git

- alternatively, you can download this repo and install all requirements by running:

		pip install -r /path/to/requirements.txt

	then to install the `pydatset` package you can run:

		python path/to/setup.py install

## Usage

Download the required dataset (e.g. cifar10 ) and call the respective `load(path)` function, for example:

	$ python
	>>> from pydatset import cifar10
	>>> data = cifar10.load('path/to/cifaf10')

Apply data augmentation to a given batch by doing something like:

	>>> from pydatset.data_augmentation import (random_contrast, random_flips,
	...                                          random_crops, random_rotate,
	...                                          random_tint)
	>>> batch = random_tint(batch)
	>>> batch = random_contrast(batch)
	>>> batch = random_flips(batch)
	>>> batch = random_rotate(batch, 10)
	>>> batch = random_crops(batch, (32, 32), pad=2)
	

check pydatset/README.md for more infos about the package contents.
