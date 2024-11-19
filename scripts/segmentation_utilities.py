#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Utilities for the aerial image segmentation
'''

#%% MODULES
import os
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import numpy as np
import re
import tifffile as tiff
from matplotlib import pyplot
from sklearn.utils import class_weight

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

def load_data(files):
    data_list = []
    for file in files:
        data = read_raster_tifffile(file)
        if data is not None:
            data_list.append(data)
    return np.array(data_list)

def read_raster_tifffile(source: str, dtype: type = np.uint8) -> np.ndarray:
    '''Reads a raster as a numpy array using Tifffile'''
    try:
        img = tiff.imread(source).astype(dtype)
        # If it's single-channel, expand dimensions to match shape conventions
        if img.ndim == 2:  # Grayscale
            img = np.expand_dims(img, axis=-1)
        return img
    except Exception as e:
        print(f"Error reading file {source}: {e}")
        return None  # Return None if there's an error


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

def calculate_class_distribution(labels):
    positive_pixels = np.sum(labels)
    total_pixels = labels.size
    negative_pixels = total_pixels - positive_pixels
    return positive_pixels, negative_pixels

def plot_class_distribution(train_dist, val_dist, test_dist):
    labels = ['Train', 'Validation', 'Test']
    positive_counts = [train_dist[0], val_dist[0], test_dist[0]]
    negative_counts = [train_dist[1], val_dist[1], test_dist[1]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = pyplot.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, positive_counts, width, label='Positive Pixels')
    rects2 = ax.bar(x + width/2, negative_counts, width, label='Negative Pixels')

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Number of Pixels')
    ax.set_title('Class Distribution in Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    pyplot.show()

def display_statistics(image_test: np.ndarray, label_test: np.ndarray, proba_predict: np.ndarray, label_predict: np.ndarray) -> None:
    '''Displays predictions statistics'''
    image_test = (image_test * 255).astype(int)
    label_test = label_test.astype(bool)
    label_predict = label_predict.astype(bool)
    
    mask_tp = np.logical_and(label_test, label_predict)
    mask_tn = np.logical_and(np.invert(label_test), np.invert(label_predict))
    mask_fp = np.logical_and(np.invert(label_test), label_predict)
    mask_fn = np.logical_and(label_test, np.invert(label_predict))
    
    colour = (255, 255, 0)  # Yellow
    masks = [mask_tp, mask_tn, mask_fp, mask_fn]
    images = [
        np.where(mask[..., np.newaxis], colour, image_test) for mask in masks
    ]
    
    images = [image_test, label_test, proba_predict, label_predict] + images
    titles = [
        'Test image', 'Test label', 'Predicted probability', 'Predicted label',
        'True positive', 'True negative', 'False positive', 'False negative'
    ]
    fig, axs = pyplot.subplots(2, 4, figsize=(20, 10))
    for image, title, ax in zip(images, titles, axs.ravel()):
        ax.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
        ax.set_title(title, fontsize=20)
        ax.axis('off')
    pyplot.tight_layout()
    pyplot.show()

#%% Metrics and Loss UTILITIES
def dice_metric(y_true, y_pred):
    """Calculate Dice Coefficient for ground truth and predicted masks."""
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    total_sum = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = tf.math.divide_no_nan(2 * intersection, total_sum)
    return dice


def iou_metric(y_true, y_pred):
    """Calculate IoU for ground truth and predicted masks."""
    y_pred = tf.cast(y_pred > 0.5, tf.float32)  # Threshold predictions at 0.5
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou = tf.math.divide_no_nan(intersection, union)  # Avoid division by zero
    return iou

def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return 1 - numerator / (denominator + tf.keras.backend.epsilon())

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice
