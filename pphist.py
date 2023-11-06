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

def get_neighbor_bins(histogram: list, index: int) -> list:
    output = []
    for i in range(index-2, index+3, 1):
        if i < 0 or i >= len(histogram):
            continue
        output.append(i)
    return output

print(imgs.keys())
nb_bins = 50
wanted_label = 8
#hist, bins = np.histogram(imgs[b'data'][0], bins=np.arange(255))

f, axarr = plt.subplots(2,1)

hist_sum = np.zeros(nb_bins)
for i, label in enumerate(imgs[b'labels'][:5000]):
    if label == wanted_label:
        hist = np.histogram(imgs[b'data'][i], bins=nb_bins)
        hist_sum += hist[0]
axarr[0].hist(hist_sum, bins=nb_bins)
hist1, bins1 = np.histogram(hist_sum, bins=nb_bins)

hist_sum = np.zeros(nb_bins)
for i, label in enumerate(imgs[b'labels'][5000:]):
    if label == wanted_label:
        hist = np.histogram(imgs[b'data'][i+5000], bins=nb_bins)
        hist_sum += hist[0]
axarr[1].hist(hist_sum, bins=nb_bins)
hist2, bins2 = np.histogram(hist_sum, bins=nb_bins)

total_diff = 0
for i, val in enumerate(hist1):
    neighbors = get_neighbor_bins(hist1, i)
    diff = 0
    for neighbor in neighbors:
        diff += abs(val - hist2[neighbor]) ** (1 / (abs(i-neighbor)+1))
    total_diff += diff
    print(val, hist2[neighbor])

print(total_diff / nb_bins)

plt.show()