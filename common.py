import matplotlib.pyplot as plt
import numpy as np
import cv2

def plot_colorful_img(img, figsize=(8, 6)):
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.grid(False)
    plt.imshow(img, interpolation='nearest', aspect='auto', cmap='Greys_r')
    
    
def normalize_img(img):
    """ Normalize image back to [0, 255] values from any other scale"""
    min, max = img.min(), img.max()
    img = np.copy(img)
    img = (img - min) / (max - min)
    return img

def clip_img(img, cast=True):
    """ Clip in range [0, 255] """
    clipped = np.clip(img, 0, 255)
    return clipped.astype(np.uint8) if cast else clipped

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def intensity_hist(img, bins=256):
    """ Histogram for image light intensity """
    plt.hist(to_gray(img).flatten(), bins=bins);
    
def color_intensity_hist(img, bins=256):
    """ Per-channel pixel intensity histogram """
    h, w, c = img.shape
    flat_img = img.reshape((-1, c))
    plt.hist(flat_img[:, 0] , bins=bins, alpha=.5, color='red')
    plt.hist(flat_img[:, 1] , bins=bins, alpha=.5, color='green')
    plt.hist(flat_img[:, 2] , bins=bins, alpha=.5, color='blue')