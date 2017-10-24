from __future__ import print_function
import pickle 
import numpy as np
import pandas as pd
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.legacy.optim as legacy_optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import scipy.misc
import scipy.ndimage
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import sys

import matplotlib.pyplot as plt
import itertools

#source of following function: http://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions
def clipped_zoom(img, zoom_factor, **kwargs):
    """
    function to zoom in or out of an image. if we zoom out the image is then padded to the original size
    """

    h, w = img.shape[:2]

    # width and height of the zoomed image
    zh = int(np.round(zoom_factor * h))
    zw = int(np.round(zoom_factor * w))

    # for multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # zooming out
    if zoom_factor < 1:
        # bounding box of the clip region within the output array
        top = (h - zh) // 2
        left = (w - zw) // 2
        # zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = scipy.ndimage.zoom(img, zoom_tuple, **kwargs)

    # zooming in
    elif zoom_factor > 1:
        # bounding box of the clip region within the input array
        top = (zh - h) // 2
        left = (zw - w) // 2
        out = scipy.ndimage.zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)
        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # if zoom_factor == 1, just return the input array
    else:
        out = img
    return out


#source of following function: https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert len(image.shape)==2

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
    
    return map_coordinates(image, indices, order=1).reshape(shape)


def data_augmentation(trainset_labeled):
    #Data augmentation
    augmented_dataset = []
    for i in trainset_labeled:
        #add original image
        augmented_dataset.append(i)


        #rotate image
        rotation_degree = np.random.uniform(-45,45)
        augmented_dataset.append((torch.from_numpy(np.array([scipy.ndimage.rotate(i[0].numpy()[0],rotation_degree,reshape=False)])),i[1]))
        

        #shift image (translation in both x and y axes)
        shift_amount = np.random.uniform(-2,2)
        augmented_dataset.append((torch.from_numpy(np.array([scipy.ndimage.shift(i[0].numpy()[0],shift_amount)])),i[1]))


        #zoom in or out of image

        #choose magnitude of amount to either zoom in or zoom out
        zoom_amount = np.random.uniform(1,2)
        
        #pick either 0 or 1, like a binary coin flip
        random_draw = np.random.randint(0,2)

        if random_draw==0: #zoom out
            zoom_amount = 1/zoom_amount
        else: #zoom in 
            zoom_amount = 1 * zoom_amount

        augmented_dataset.append((torch.from_numpy(np.array([clipped_zoom(i[0].numpy()[0],zoom_amount)])),i[1]))


        #apply elastic distortion to image
        #paramter values based off of https://arxiv.org/pdf/1103.4487.pdf
        sigma = np.random.uniform(5,6)
        alpha = np.random.uniform(36,38)
        augmented_dataset.append((torch.from_numpy(np.array([elastic_transform(i[0].numpy()[0],alpha,sigma)])),i[1]))

    return augmented_dataset