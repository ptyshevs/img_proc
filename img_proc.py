import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from common import *
from quantizers import *
from misc_func import *


def read_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def save(img, path, prefix=''):
    parts = path.split(os.path.sep)
    parts = parts[:-1] + [prefix + '_' + parts[-1]]
    path_with_prefix = os.path.join(*parts)
    print("Path to save", path_with_prefix)
    cv2.imwrite(path_with_prefix, img)

def image_stats(img):
    return img.shape

def img_cnt_colors(img):
    h, w, c = img.shape
    return len(np.unique(img.reshape((-1, c), axis=0)))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('img', help="Image to process")
    parser.add_argument('--stat', '-s', help="Show shape of image", action='store_true', default=False)
    parser.add_argument('--gray', '-g', help="Turn colorful image into grayscale", action='store_true', default=False)
    parser.add_argument('--quantize_unif_gray', help="Uniform quantization of grayscale image", action='store_true', default=False)
    parser.add_argument('--quantize_unif_color', help="Uniform quantization of RGB image", action='store_true', default=False)
    parser.add_argument('--quantize_knn', help="Quantization using k nearest neighbors", action='store_true', default=False)
    parser.add_argument('--quantize_median', help="Quantization using median filter", action='store_true', default=False)
    parser.add_argument('--quantization_levels', default=16, help='Number of unique colors to use when quantizing image')
    parser.add_argument('--threshold', '-t', help="Use binary thresholding", action='store_true', default=False)
    parser.add_argument('--white_noise', '-w', help="Add white noise to the image", action='store_true', default=False)
    parser.add_argument('--dither', '-d', help="Dither the image", action='store_true', default=False)
    parser.add_argument('--bayes_filter', '-bf', help="Bayes filter", action='store_true', default=False)
    parser.add_argument('--gauss_filter', '-gf', help="Gauss filter", action='store_true', default=False)
    parser.add_argument('--sobel', '-so', help="Sobel filter", action='store_true', default=False)
    parser.add_argument('--prewitt', '-p', help="Prewitt filter", action='store_true', default=False)
    parser.add_argument('--laplace', '-l', help="Laplace filter", action='store_true', default=False)

    parser.add_argument('--save_to', default='result.png', help="Path to where save image")

    args = parser.parse_args()
    img = read_img(args.img)
    if args.stat:
        print("Shape:", image_stats(img))
        print(f"img size: {len(img) * img.shape[1]}, unique colors: {img_cnt_colors(img)}")
    
    if args.gray:
        img = to_gray(img)
    
    prefix = ''
    res = None
    if args.quantize_unif_gray:
        prefix = 'quantize_unif_gray'
        res = quantize_unif_gray(img, n_levels=args.quantization_levels)
    elif args.quantize_unif_color:
        prefix = 'quantize_unif_color'
        res = quantize_unif_color(img, n_levels=args.quantization_levels)
    elif args.quantize_knn:
        prefix = 'quantize_knn'
        res = knn_quantizer(img, levels=args.quantization_levels)
    elif args.quantize_median:
        prefix = 'quantize_median'
        res = median_quantizer(img, max_depth=int(np.log2(args.quantization_levels)))
    elif args.threshold:
        prefix = 'threshold'
        res = threshold_img(img)
    elif args.white_noise:
        prefix = 'white_noise'
        res = add_white_noise_colorful(img) if len(img.shape) == 3 else add_white_noise_gray(img)
    elif args.dither:
        prefix = 'dither'
        res = dither_floyd_steinberg(img)
    elif args.bayes_filter:
        prefix = 'bayes_filter'
        res = bayer_filter(img)
    elif args.gauss_filter:
        prefix = 'gauss_filter'
        res = gauss_filter(img)
    elif args.sobel:
        prefix = 'sobel'
        res = sobel(img)
    elif args.prewitt:
        prefix = 'prewitt'
        res = prewitt(img)
    elif args.laplace:
        prefix = 'laplace'
        res = laplace_operator(img)
    elif args.gray:
        prefix = 'gray'
        res = to_gray(img)

    save(res, args.save_to, prefix=prefix)