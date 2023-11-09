import matplotlib.pyplot as plt
import cv2
import numpy as np
import math

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

imgs = unpickle("../cifar-10-python/cifar-10-batches-py/data_batch_1")

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

print(imgs.keys())
nb_bins = 25
wanted_label = 8
#hist, bins = np.histogram(imgs[b'data'][0], bins=np.arange(255))

f, axarr = plt.subplots(2,1)

hist1 = np.zeros(nb_bins)
c = 0
for i, label in enumerate(imgs[b'labels'][:5000]):
    if label == wanted_label:
        gray = to_grayscale(imgs[b'data'][i])
        hist = np.histogram(gray, bins=nb_bins)
        hist1 += hist[0]
        c += 1

hist1 = np.divide(hist1, c)
axarr[0].hist(hist1, bins=nb_bins)


hist2 = np.zeros(nb_bins)
c = 0
for i, label in enumerate(imgs[b'labels'][5000:]):
    if label == wanted_label:
        gray = to_grayscale(imgs[b'data'][i])
        hist = np.histogram(gray, bins=nb_bins)
        hist2 += hist[0]
        c += 1

hist2 = np.divide(hist2, c)
axarr[1].hist(hist2, bins=nb_bins)


total_diff = 0
for i, val in enumerate(hist1):
    neighbor = get_similar_neighbor_in_range(hist1, hist2, i, 2)
    
    total_diff += abs(val - hist2[neighbor])

print(math.sqrt(total_diff / nb_bins))

plt.show()