import matplotlib.pyplot as plt
import cv2
import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

imgs = unpickle("cifar-10-python/cifar-10-batches-py/data_batch_1")

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

hist_sum = np.zeros(nb_bins)
for i, label in enumerate(imgs[b'labels'][5000:]):
    if label == wanted_label:
        hist = np.histogram(imgs[b'data'][i+5000], bins=nb_bins)
        hist_sum += hist[0]
axarr[1].hist(hist_sum, bins=nb_bins)

plt.show()