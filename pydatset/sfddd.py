#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
from scipy.ndimage import imread


def get_data(directory, num_validation=2000):
    '''
    Load the SFDDD dataset from disk and perform preprocessing to prepare
    it for the neural net classifier.
    '''
    # Load the raw SFDDD data
    Xtr, Ytr = load(directory)

    X_test = Xtr[:num_validation]
    y_test = Ytr[:num_validation]

    X_train = Xtr[num_validation:]
    y_train = Ytr[num_validation:]

    l = len(X_train)
    mask = np.random.choice(l, l)
    X_train = X_train[mask]
    y_train = y_train[mask]

    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2)
    X_test = X_test.transpose(0, 3, 1, 2)

    mean_image = np.mean(X_train, axis=0)
    std = np.std(X_train)

    X_train -= mean_image
    X_test -= mean_image

    X_train /= std
    X_test /= std

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_test': X_test, 'y_test': y_test,
        'mean': mean_image, 'std': std
    }


def load_imgs(folder):
    '''
    Load all images in a folder
    '''
    names = [x for x in os.listdir(folder) if '.jpg' in x]
    num_of_images = len(names)
    imgs = []
    for i in range(num_of_images):
        imgs.append(imread(os.path.join(folder, names[i]), mode='RGB'))
    return imgs


def load(ROOT):
    ''' load all of SFDDD '''
    xs = []
    ys = []
    for b in range(10):
        imgs = load_imgs(os.path.join(ROOT, 'train', 'c%d' % (b, )))
        ys.append([b] * len(imgs))
        xs.append(imgs)

    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)

    mask = np.arange(len(Ytr))
    np.random.shuffle(mask)
    Xtr = Xtr[mask]
    Ytr = Ytr[mask]

    return Xtr, Ytr
