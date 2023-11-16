import matplotlib.pyplot as plt
import numpy as np
import math
import random
from collections.abc import Callable
from histograms import Summary

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# todo remove found neighbor
def get_similar_neighbor_in_range(histogram1: list, histogram2: list, index: int, wideness: int) -> int:
    best_i, best = 0, 255
    for i in range(index-wideness, index+wideness, 1):
        if i < 0 or i >= len(histogram1):
            continue
        diff = abs(histogram1[i] - histogram2[i])
        if diff < best:
            best_i = i
            best = diff
    return best_i

# input:  list sized 3072 (1024 * rgb)
# output: list sized 1024 (grayscale)
def to_grayscale(image: list) -> list:
    output = [0] * 1024
    for i in range(1024):
        output[i] = int(image[i] * 0.299 + image[i+1024] * 0.587 + image[i+2048] * 0.114)
    return output

def extract_images_from_label(imgs: dict, wanted_label: str) -> list:
    matching_imgs = []
    for i, label in enumerate(imgs[b'labels']):
        if label == wanted_label:
            matching_imgs.append(imgs[b'data'][i])
    return matching_imgs

def summarize_histogram(histogram: list) -> Summary:
    return Summary(histogram)

def randomize_histogram(histogram: list, e: float) -> list:
    return list(map(lambda x: x + random.uniform(-e, e), histogram))

def histogram_from_grayscale(img: list, nb_bins: int) -> list:
    gray = to_grayscale(img)
    return np.histogram(gray, bins=nb_bins)[0]

def histogram_from_contrast(img: list, nb_bins: int) -> list:
    gray = to_grayscale(img)
    hist = np.histogram(gray, bins=nb_bins)
    return np.cumsum(hist[0])

def images_to_histogram(imgs: list, transform_func: Callable[[list, int], list], nb_bins: int) -> list:
    hist_sum = np.zeros(nb_bins)
    c = 0
    for img in imgs:
        hist = transform_func(img, nb_bins)
        hist_sum += hist
        c += 1
    hist_sum = np.divide(hist_sum, c)
    return hist_sum

def compare_histogram(hist1: list, hist2: list) -> int:
    total_diff = 0
    for i, val in enumerate(hist1):
        neighbor = get_similar_neighbor_in_range(hist1, hist2, i, 2)
    
        total_diff += abs(val - hist2[neighbor])
    return math.sqrt(total_diff / len(hist1))

if __name__ == '__main__':
    dataset = unpickle("../cifar-10-python/cifar-10-batches-py/data_batch_1")
    nb_bins = 255
    wanted_label = 8