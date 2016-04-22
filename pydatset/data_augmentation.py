#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from skimage.transform import rotate
from skimage.util import random_noise



def random_flips(X):
    '''
    Take random x-y flips of images.

    Input:
    - X: (N, C, H, W) array of image data.

    Output:
    - An array of the same shape as X, containing a copy of the data in X,
      but with half the examples flipped along the horizontal direction.
    '''

    N, C, H, W = X.shape
    mask = np.random.randint(2, size=N)
    out = np.zeros_like(X)
    out[mask == 1] = X[mask == 1, :, :, ::-1]
    out[mask == 0] = X[mask == 0]
    return out


def add_pad(X, pad):
    '''
    Take random crops of images. For each input image we will generate a random
    crop of that image of the specified size.

    Input:
    - X: (N, C, H, W) array of image data
    - pad: Number of white pixels to add on each side of each image

    Output:
    - Array of shape (N, C, H + 2 * pad, WW + 2 * pad)
    '''
    N, C, H, W = X.shape
    assert pad > 0

    out = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    return out


def random_crops(X, crop_shape, pad=0):
    '''
    Take random crops of images. For each input image we will generate a random
    crop of that image of the specified size.

    Input:
    - X: (N, C, H, W) array of image data
    - crop_shape: Tuple (HH, WW) to which each image will be cropped.

    Output:
    - Array of shape (N, C, HH, WW)
    '''
    if pad:
        X = add_pad(X, pad)
    N, C, H, W = X.shape
    HH, WW = crop_shape
    assert HH < H and WW < W

    out = np.zeros((N, C, HH, WW), dtype=X.dtype)

    np.random.randint((H-HH), size=N)
    y_start = np.random.randint((H-HH), size=N)
    x_start = np.random.randint((W-WW), size=N)

    for i in xrange(N):
        out[i] = X[i, :, y_start[i]:y_start[i]+HH, x_start[i]:x_start[i]+WW]

    return out


def random_rotate(X, max_angle=10):
    N, C, H, W = X.shape
    out = np.zeros_like(X)
    high = np.abs(max_angle) + 1
    low = - np.abs(max_angle)
    for i, x in enumerate(X):
        t = x.transpose(1, 2, 0)
        t = rotate(t, np.random.randint(low, high), resize=False)
        t = t.transpose(2, 0, 1)

        out[i] = t
    return out


def random_contrast(X, scale=(0.8, 1.2)):
    '''
    Randomly adjust the contrast of images. For each input image, choose a
    number uniformly at random from the range given by the scale parameter,
    and multiply each pixel of the image by that number.
    source:
    https://github.com/MyHumbleSelf/cs231n/blob/master/assignment3/cs231n/data_augmentation.py

    Inputs:
    - X: (N, C, H, W) array of image data
    - scale: Tuple (low, high). For each image we sample a scalar in the
      range (low, high) and multiply the image by that scaler.

    Output:
    - Rescaled array out of shape (N, C, H, W) where out[i] is a contrast
      adjusted version of X[i].
    '''
    low, high = scale
    N = X.shape[0]
    out = np.zeros_like(X)
    l = (scale[1]-scale[0])*np.random.random_sample(N)+scale[0]
    # for i in xrange(N):
    #   out[i] = X[i] * l[i]
    out = X * l[:, None, None, None]
    # TODO: vectorize this somehow...
    # out =   #np.diag(l).dot(X)#X*l[:,np.newaxis, np.newaxis, np.newaxis]

    return out


def random_tint(X, scale=(-10, 10)):
    '''
    Randomly tint images. For each input image, choose a random color whose
    red, green, and blue components are each drawn uniformly at random from
    the range given by scale. Add that color to each pixel of the image.
    source:
    https://github.com/MyHumbleSelf/cs231n/blob/master/assignment3/cs231n/data_augmentation.py

    Inputs:
    - X: (N, C, W, H) array of image data
    - scale: A tuple (low, high) giving the bounds for the random color that
      will be generated for each image.

    Output:
    - Tinted array out of shape (N, C, H, W) where out[i] is a tinted version
      of X[i].
    '''
    low, high = scale
    N, C = X.shape[:2]
    out = np.zeros_like(X)

    l = (scale[1]-scale[0])*np.random.random_sample((N, C))+scale[0]
    out = X+l[:, :, None, None]

    return out


def fixed_crops(X, crop_shape, crop_type):
    '''
    Take center or corner crops of images.
    source:
    https://github.com/MyHumbleSelf/cs231n/blob/master/assignment3/cs231n/data_augmentation.py

    Inputs:
    - X: Input data, of shape (N, C, H, W)
    - crop_shape: Tuple of integers (HH, WW) giving the size to which each
      image will be cropped.
    - crop_type: One of the following strings, giving the type of crop to
      compute:
      'center': Center crop
      'ul': Upper left corner
      'ur': Upper right corner
      'bl': Bottom left corner
      'br': Bottom right corner

    Returns:
    Array of cropped data of shape (N, C, HH, WW)
    '''
    N, C, H, W = X.shape
    HH, WW = crop_shape

    x0 = (W - WW) / 2
    y0 = (H - HH) / 2
    x1 = x0 + WW
    y1 = y0 + HH

    if crop_type == 'center':
        return X[:, :, y0:y1, x0:x1]
    elif crop_type == 'ul':
        return X[:, :, :HH, :WW]
    elif crop_type == 'ur':
        return X[:, :, :HH, -WW:]
    elif crop_type == 'bl':
        return X[:, :, -HH:, :WW]
    elif crop_type == 'br':
        return X[:, :, -HH:, -WW:]
    else:
        raise ValueError('Unrecognized crop type %s' % crop_type)


def RGB_PCA(images):
    '''
    Source: https://github.com/Thrandis/ift6266h15/blob/1cc3fc6164dc6c54936971
    935027cd447e2cd81f/dataset_augmentation.py

    RGB PCA and variations from Alex's paper
     '''
    pixels = images.reshape(-1, images.shape[-1])
    idx = np.random.random_integers(0, pixels.shape[0], 1000000)
    pixels = [pixels[i] for i in idx]
    pixels = np.array(pixels, dtype=np.uint8).T
    m = np.mean(pixels)/256.
    C = np.cov(pixels)/(256.*256.)
    l, v = np.linalg.eig(C)
    return l, v, m


def RGB_variations(image, eig_val, eig_vec):
    '''
    Source: https://github.com/Thrandis/ift6266h15/blob/1cc3fc6164dc6c54936971
    935027cd447e2cd81f/dataset_augmentation.py
     '''
    a = np.random.randn(3)
    v = np.array([a[0]*eig_val[0], a[1]*eig_val[1], a[2]*eig_val[2]])
    variation = np.dot(eig_vec, v)
    return image + variation


def noise(x):
    '''
    Source: https://github.com/Thrandis/ift6266h15/blob/1cc3fc6164dc6c54936971
    935027cd447e2cd81f/dataset_augmentation.py
     '''
    r = np.random.rand(1)[0]
    # TODO randomize parameters of the noises; check how to init seed
    if r < 0.33:
        return random_noise(x, 's&p', seed=np.random.randint(1e6))
    if r < 0.66:
        return random_noise(x, 'gaussian', seed=np.random.randint(1e6))
    return random_noise(x, 'speckle', seed=np.random.randint(1e6))


def create_2d_gaussian(dim, sigma):
    """
    Source: https://github.com/vsvinayak/mnist-helper/blob/master/mnist_helpers.py
    This function creates a 2d gaussian kernel with the standard deviation
    denoted by sigma

    :param dim: integer denoting a side (1-d) of gaussian kernel
    :type dim: int
    :param sigma: the standard deviation of the gaussian kernel
    :type sigma: float

    :returns: a numpy 2d array
    """

    # check if the dimension is odd
    if dim % 2 == 0:
        raise ValueError("Kernel dimension should be odd")

    # initialize the kernel
    kernel = np.zeros((dim, dim), dtype=np.float16)

    # calculate the center point
    center = dim/2

    # calculate the variance
    variance = sigma ** 2

    # calculate the normalization coefficeint
    coeff = 1. / (2 * variance)

    # create the kernel
    for x in range(0, dim):
        for y in range(0, dim):
            x_val = abs(x - center)
            y_val = abs(y - center)
            numerator = x_val**2 + y_val**2
            denom = 2*variance

            kernel[x, y] = coeff * np.exp(-1. * numerator/denom)

    # normalise it
    return kernel/sum(sum(kernel))


def elastic_transform(image, kernel_dim=13, sigma=6, alpha=36, negated=False):
    """
    Source: https://github.com/vsvinayak/mnist-helper/blob/master/mnist_helpers.py
    This method performs elastic transformations on an image by convolving
    with a gaussian kernel.
    NOTE: Image dimensions should be a sqaure image

    :param image: the input image
    :type image: a numpy nd array
    :param kernel_dim: dimension(1-D) of the gaussian kernel
    :type kernel_dim: int
    :param sigma: standard deviation of the kernel
    :type sigma: float
    :param alpha: a multiplicative factor for image after convolution
    :type alpha: float
    :param negated: a flag indicating whether the image is negated or not
    :type negated: boolean
    :returns: a nd array transformed image
    """
    # TEMP FIX
    import math

    from numpy.random import random_integers
    from scipy.signal import convolve2d
    import cv2

    # convert the image to single channel if it is multi channel one
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # check if the image is a negated one
    if not negated:
        image = 255-image

    # check if the image is a square one
    if image.shape[0] != image.shape[1]:
        raise ValueError("Image should be of sqaure form")

    # check if kernel dimesnion is odd
    if kernel_dim % 2 == 0:
        raise ValueError("Kernel dimension should be odd")

    # create an empty image
    result = np.zeros(image.shape)

    # create random displacement fields
    displacement_field_x = np.array([[random_integers(-1, 1) for x in xrange(image.shape[0])]
                                     for y in xrange(image.shape[1])]) * alpha
    displacement_field_y = np.array([[random_integers(-1, 1) for x in xrange(image.shape[0])]
                                     for y in xrange(image.shape[1])]) * alpha

    # create the gaussian kernel
    kernel = create_2d_gaussian(kernel_dim, sigma)

    # convolve the fields with the gaussian kernel
    displacement_field_x = convolve2d(displacement_field_x, kernel)
    displacement_field_y = convolve2d(displacement_field_y, kernel)

    # make the distortrd image by averaging each pixel value to the neighbouring
    # four pixels based on displacement fields

    for row in xrange(image.shape[1]):
        for col in xrange(image.shape[0]):
            low_ii = row + int(math.floor(displacement_field_x[row, col]))
            high_ii = row + int(math.ceil(displacement_field_x[row, col]))

            low_jj = col + int(math.floor(displacement_field_y[row, col]))
            high_jj = col + int(math.ceil(displacement_field_y[row, col]))

            if low_ii < 0 or low_jj < 0 or high_ii >= image.shape[1] - 1 \
               or high_jj >= image.shape[0] - 1:
                continue

            res = image[low_ii, low_jj]/4 + image[low_ii, high_jj]/4 + \
                image[high_ii, low_jj]/4 + image[high_ii, high_jj]/4

            result[row, col] = res

    # if the input image was not negated, make the output image also a non
    # negated one
    if not negated:
        result = 255-result

    return result
