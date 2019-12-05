import numpy as np
import matplotlib.pyplot as plt
from common import *
import cv2

def mse(a, b):
    return ((a - b) ** 2).sum()

def quantize_unif_gray(img, n_levels=10):
    gray = to_gray(img)
    gray_min, gray_max = gray.min(), gray.max()
    dynamic_range = gray_max - gray_min
    q = dynamic_range / n_levels
    img_quantized = np.floor((gray - gray_min) / q) * q + q/2 + gray_min
    print(f"Dynamic range={dynamic_range} | q={q} | mse={mse(gray, img_quantized)}")
    return img_quantized

def quantize_unif_color(img, n_levels=10):
    f_min = img.min(axis=0).min(axis=0)
    f_max = img.max(axis=0).max(axis=0)
    dynamic_range = f_max - f_min
    q = dynamic_range / n_levels
    print(f"Dynamic range={dynamic_range} | q={q}")#" | mse={mse(gray, img_quantized)}")

    img_quantized = np.clip(np.floor((img - f_min) / q) * q + q / 2 + f_min, 0, 255)
    print(img_quantized.min(), img_quantized.mean(), img_quantized.max())
    return img_quantized.astype(int)

def knn_quantizer(img, k=3, levels=16, interpolate=True, verbose=False):
    """
    Basically, vector quantization performed for single image.
    If you have many images, this becomed LVQ, well-known supervised dim.red. method.
    
    At first glance, it seems like we don't win any space by replacing colors with most common ones.
    
    @param img: RGB img of shape height x width x n_channels
    """
    # Step 1: Find top-[levels] most frequent colors
    h, w, c = img.shape
    flat_img = np.copy(img.reshape((-1, c))).astype(np.float64)
    unique, unique_counts = np.unique(flat_img, axis=0, return_counts=True)
    freq_idx = np.argsort(unique_counts)[::-1][:levels]
    freq_colors = unique[freq_idx]
    freq_counts = unique_counts[freq_idx]
    
    if verbose:
        print(f"{len(freq_colors)} colors selected, most common one is {freq_colors[0]} used in {freq_counts[0]} pixels")
    
    # Step 2: Calculate Euclidean distance between each unique color and most frequent ones
    dist = lambda x: np.sqrt(np.linalg.norm(x, ord=2))
    # Matrix of shape n_unique_colors x levels
    distances = np.apply_along_axis(lambda x: np.array([dist(fc - x) for fc in freq_colors]), 1, unique)
    
    if verbose:
        print("Finished building distances matrix")
    idx = np.argsort(distances)  # Each row is sorted by distance to freq_colors
    
    top_idx = idx[:, :k]
    color_map = np.zeros_like(unique, dtype=np.float64)
    # Step 3: Use linear interpolation of neighboring colors. Basically, weight each color by distance to it
    if interpolate:
        for i, idx_row in tqdm.tqdm(enumerate(top_idx)):
            closest_distances = distances[i][idx_row]
            closest_colors = freq_colors[idx_row]
            interpolated = (closest_colors.T @ closest_distances) / closest_distances.sum()
            color_map[i] = interpolated
    else:
        color_map = freq_colors[top_idx[:, 0]]
    
    if verbose:
        print("Finished interpolating colors")
    # Step 4: Now we can replace each unique color by it's interpolated quantized value
    for i, uc in tqdm.tqdm(enumerate(unique)):
        flat_img = np.where(flat_img == uc, color_map[i], flat_img)
    if verbose:
        print("Finished quantizing")
    return flat_img.reshape((h, w, c))

def find_split_axis(data):
    """ @param data: n_pixels x n_channels """
    fmin = np.min(data, axis=0)
    fmax = np.max(data, axis=0)
    dynamic_range = fmax - fmin
    split_axis = np.argmax(dynamic_range)
    return split_axis, dynamic_range

def split_by_axis(data, split_axis):
    sorted_data = data[np.argsort(data[:, split_axis])]
    split_median = np.median(sorted_data[:, split_axis])
    left_bin = sorted_data[sorted_data[:, split_axis] < split_median]
    right_bin = sorted_data[sorted_data[:, split_axis] >= split_median]
    return split_median, left_bin, right_bin

class Node:
    def __init__(self, data, axis=None, median=None, left=None, right=None):
        self.data = data
        self.axis = axis
        self.median = median
        self.fill_median = np.median(data, axis=0)
        self.left = left
        self.right = right
    
    @property
    def is_leaf(self):
        return not self.left and not self.right
    
    def predict(self, x, verbose=False):
        if verbose:
            print(f"predict called on {self.data.shape}, is_leaf:", self.is_leaf, 'median:', self.median, 'axis:', self.axis)
        if self.is_leaf:
            if verbose:
                print("leaf, predicting:", np.median(self.data, axis=0))
            return np.median(self.data, axis=0)
        predictions = np.zeros_like(x)
        predictions[:] = self.fill_median
        idx = x[..., self.axis] < self.median
        if self.left:
            predictions[idx] = self.left.predict(x[idx,:], verbose=verbose)
        if self.right:
            predictions[~idx] = self.right.predict(x[~idx, :], verbose=verbose)
        return predictions

def median_quantizer(img, max_depth=5, verbose=False):
    """ Max unique levels = 2 ** max_depth """
    h, w, c = img.shape
    flat_img = img.reshape((-1, c))
    tree = None
    data = flat_img
    verbose = False
    min_sample_split = 4

    split_queue = [(0, flat_img, None)]
    
    while split_queue:
        cur_depth, data, cur_node = split_queue.pop(0)

        if cur_depth >= max_depth or len(data) < min_sample_split:
            continue
        if verbose:
            print(f"depth {cur_depth}")

        split_axis, dynamic_range = find_split_axis(data)
        median, left_data, right_data = split_by_axis(data, split_axis)
        left_node = Node(left_data) if len(left_data) else None
        right_node = Node(right_data) if len(right_data) else None
        if verbose:
            print(f"depth={cur_depth}, dyn.range={dynamic_range}, split by {split_axis}, median={median}, cur_size={len(data)}, left_size={len(left_data)}, right_size={len(right_data)}, median_left={np.median(left_data, axis=0)}, median_right={np.median(right_data,axis=0)}")
        if cur_node is None:
            tree = Node(data, split_axis, median, left_node, right_node)
        else:
            cur_node.axis = split_axis
            cur_node.median = median
            cur_node.left = left_node
            cur_node.right = right_node
        if len(left_data) > min_sample_split:
            split_queue.append((cur_depth + 1, left_data, left_node))
        if len(right_data) > min_sample_split:
            split_queue.append((cur_depth + 1, right_data, right_node))
    
    return tree.predict(flat_img).reshape((h, w, c))