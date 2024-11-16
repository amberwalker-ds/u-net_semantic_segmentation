#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Utilities for the aerial image segmentation homework
'''

#%% MODULES

import numpy as np
import os
import rasterio
import re

from matplotlib import pyplot

#%% FILE UTILITIES

def search_files(directory:str, pattern:str='.') -> list:
    '''Searches files in a directory'''
    files = list()
    for root, _, file_names in os.walk(directory):
        for file_name in file_names:
            files.append(os.path.join(root, file_name))
    files = list(filter(re.compile(pattern).search, files))
    files.sort()
    return files

#%% RASTER UTILITIES

def read_raster(source:str, dtype:type=np.uint8) -> np.ndarray:
    '''Reads a raster as a numpy array'''
    raster = rasterio.open(source)
    raster = raster.read()
    image  = raster.transpose([1, 2, 0]).astype(dtype)
    return image

def write_raster(image:np.ndarray, source:str, destination:str, nodata:int=0, dtype:str='uint8') -> None:
    '''Writes a numpy array as a raster'''
    raster  = image.transpose([2, 0, 1]).astype(dtype)
    profile = rasterio.open(source).profile
    profile.update(dtype=dtype, count=raster.shape[0], nodata=nodata)
    with rasterio.open(destination, 'w', **profile) as dest:
        dest.write(raster)

#%% DISPLAY UTILITIES

def display(image:np.ndarray, title:str='') -> None:
    '''Displays an image'''
    fig, ax = pyplot.subplots(1, figsize=(5, 5))
    ax.imshow(image, cmap='gray')
    ax.set_title(title, fontsize=20)
    ax.set_axis_off()
    pyplot.tight_layout()
    pyplot.show()

def compare(images:list, titles:list=['']) -> None:
    '''Displays multiple images'''
    nimage = len(images)
    if len(titles) == 1:
        titles = titles * nimage
    fig, axs = pyplot.subplots(nrows=1, ncols=nimage, figsize=(5, 5*nimage))
    for ax, image, title in zip(axs.ravel(), images, titles):
        ax.imshow(image, cmap='gray')
        ax.set_title(title, fontsize=15)
        ax.set_axis_off()
    pyplot.tight_layout()
    pyplot.show()

def display_history(history:dict, metrics:list=['accuracy', 'loss']) -> None:
    '''Displays training history'''
    fig, axs = pyplot.subplots(nrows=1, ncols=2, figsize=(10, 5))
    for ax, metric in zip(axs.ravel(), metrics):
        ax.plot(history[metric])
        ax.plot(history[f'val_{metric}'])
        ax.set_title(f'Training {metric}', fontsize=15)
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Epoch')
        ax.legend(['Training sample', 'Validation sample'], frameon=False)
    pyplot.tight_layout()
    pyplot.show()

def display_statistics(image_test:np.ndarray, label_test:np.ndarray, proba_predict:np.ndarray, label_predict:np.ndarray) -> None:
    '''Displays predictions statistics'''
    # Format arrays
    image_test    = (image_test * 255).astype(int)
    label_test    = label_test.astype(bool)
    label_predict = label_predict.astype(bool)
    # Computes statistics
    mask_tp = np.logical_and(label_test, label_predict)
    mask_tn = np.logical_and(np.invert(label_test), np.invert(label_predict))
    mask_fp = np.logical_and(np.invert(label_test), label_predict)
    mask_fn = np.logical_and(label_test, np.invert(label_predict))
    # Augments images
    colour  = (255, 255, 0)
    images  = [np.where(np.tile(mask, (1, 1, 3)), colour, image_test) for mask in [mask_tp, mask_tn, mask_fp, mask_fn]]
    # Figure
    images = [image_test, label_test, proba_predict, label_predict] + images
    titles = ['Test image', 'Test label', 'Predicted probability', 'Predicted label', 'True positive', 'True negative', 'False positive', 'False negative']
    fig, axs = pyplot.subplots(2, 4, figsize=(20, 10))
    for image, title, ax in zip(images, titles, axs.ravel()):
        ax.imshow(image)
        ax.set_title(title, fontsize=20)
        ax.axis('off')
    pyplot.tight_layout()
    pyplot.show()