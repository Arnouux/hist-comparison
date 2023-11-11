import matplotlib.pyplot as plt
import cv2
import numpy as np
import math

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

dataset = unpickle("../cifar-10-python/cifar-10-batches-py/data_batch_1")

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

def images_to_histogram(imgs: list, nb_bins: int) -> list:
    hist1 = np.zeros(nb_bins)
    c = 0
    for img in imgs:
        gray = to_grayscale(img)
        hist = np.histogram(gray, bins=nb_bins)
        hist1 += hist[0]
        c += 1
    hist1 = np.divide(hist1, c)
    return hist1

nb_bins = 25
wanted_label = 8
f, axarr = plt.subplots(2,1)

imgs = extract_images_from_label(dataset, wanted_label)
middle = len(imgs)//2
batch1 = imgs[:middle]
batch2 = imgs[middle:]

print(len(imgs))

hist1 = images_to_histogram(batch1, nb_bins)
axarr[0].hist(hist1, bins=nb_bins)

hist2 = images_to_histogram(batch2, nb_bins)
axarr[1].hist(hist2, bins=nb_bins)

total_diff = 0
for i, val in enumerate(hist1):
    neighbor = get_similar_neighbor_in_range(hist1, hist2, i, 2)
    
    total_diff += abs(val - hist2[neighbor])

print(math.sqrt(total_diff / nb_bins))

plt.show()