from common import *
import numpy as np
import cv2
import scipy
import scipy.signal

def threshold_img(img, low=3, high=250):
    gray_img = to_gray(img)
    return np.where((gray_img >= low) & (gray_img <= high), 1., 0.)

def add_white_noise_gray(img, scale=10):
    img = to_gray(img)
    h, w = img.shape
    return np.clip(img + np.random.randn(h, w) * scale, 0, 255).astype(np.uint8)

def add_white_noise_colorful(img, scale=10):
    h, w, c = img.shape
    return np.clip(img + np.random.randn(h, w, c) * scale, 0, 255).astype(np.uint8)

def dither_floyd_steinberg(img, n_levels=16, error_scattering=True):
    h, w, c = img.shape
    f_min = img.min(axis=0).min(axis=0)
    f_max = img.max(axis=0).max(axis=0)
    dynamic_range = f_max - f_min
    q = dynamic_range / n_levels
    print(f"Dynamic range={dynamic_range} | q={q}")#" | mse={mse(gray, img_quantized)}")

    img_quantized = np.floor((img - f_min) / q) * q + q / 2 + f_min
    if not error_scattering:
        return np.clip(img_quantized, 0, 255).astype(np.uint8)
    for i in range(h):
        for j in range(w):
            err = (img[i, j] - img_quantized[i, j]).astype(float) / 16
            if j < w - 1:
                img_quantized[i, j+1] += err * 7
            if i < h - 1:
                img_quantized[i + 1, j] += err * 5
                if j > 0:
                    img_quantized[i + 1, j-1] += err * 3
                if j < w - 1:
                    img_quantized[i + 1, j + 1] += err
    return np.clip(img_quantized, 0, 255).astype(np.uint8)

def bayer_filter(img):
    """ Extract values from RGB images according to Bayer filter"""
    res = np.zeros_like(img)
    h, w, c = img.shape
    for i in range(0, h, 2):
        for j in range(0, w, 2):
            res[i, j] = img[i, j, 1]
            if j + 1 < w - 1:
                res[i, j+1] = img[i, j+1, 0]
            if i + 1 < h - 1:
                res[i + 1, j] = img[i + 1, j, 2]
            if j + 1 < w - 1 and i + 1 < h - 1:
                res[i + 1, j + 1] = img[i + 1, j + 1, 1]
    return res

def gauss_filter(img, size=3, runs=1):
    if size & 1 != 1:
        raise ValueError("Gauss kernel can only be of odd size")
    if size == 3:
        kernel = np.array([[1, 2, 1],
                         [2, 4, 2],
                         [1, 2, 1]])
    elif size == 5:
        kernel = np.array([[1, 4, 7, 4, 1],
                           [4, 16, 26, 16, 4],
                           [7, 26, 41, 26, 7],
                           [4, 16, 26, 16, 4],
                           [1, 4, 7, 4, 1]])
    normalized_kernel = kernel / kernel.sum()
    if len(img.shape) == 3:
        normalized_kernel = normalized_kernel[...,np.newaxis]
    res = np.copy(img)
    for i in range(runs):
        res = scipy.signal.fftconvolve(res, normalized_kernel, mode='same')
    return res

def sobel(img, grey=True):
    if grey:
        img = to_gray(img)
    dx_kernel = np.array([[-1, -2, -1],
                          [0, 0, 0],
                          [1, 2, 1]])
    dy_kernel = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]])
    
    if len(img.shape) == 3:
        dx_kernel = dx_kernel[..., np.newaxis]
        dy_kernel = dy_kernel[..., np.newaxis]
    
    dx = scipy.signal.fftconvolve(img, dx_kernel, mode='same')
    dy = scipy.signal.fftconvolve(img, dy_kernel, mode='same')
    g = np.sqrt(dx ** 2 + dy ** 2)
    return np.clip(g, 0, 255).astype(np.uint8)

def prewitt(img, grey=True):
    if grey:
        img = to_gray(img)
    dx_kernel = np.array([[-1, -1, -1],
                          [0, 0, 0],
                          [1, 1, 1]])
    dy_kernel = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]])
    
    if len(img.shape) == 3:
        dx_kernel = dx_kernel[..., np.newaxis]
        dy_kernel = dy_kernel[..., np.newaxis]
    
    dx = scipy.signal.fftconvolve(img, dx_kernel, mode='same')
    dy = scipy.signal.fftconvolve(img, dy_kernel, mode='same')
    g = np.sqrt(dx ** 2 + dy ** 2)
    return np.clip(g, 0, 255).astype(np.uint8)

def laplace_operator(img):
    laplacian = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]])
    if len(img.shape) == 3:
        laplacian = laplacian[...,np.newaxis]
    
    return scipy.signal.fftconvolve(img, laplacian, mode='same')