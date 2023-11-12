import unittest
from pphist import *

class TestHistogramComparison(unittest.TestCase):

    def setUp(self):
        dataset = unpickle("../cifar-10-python/cifar-10-batches-py/data_batch_1")
        self.small_dataset1, self.small_dataset2 = {}, {}
        self.small_dataset1[b'labels'] = dataset[b'labels'][500:1000]
        self.small_dataset1[b'data'] = dataset[b'data'][500:1000]
        self.small_dataset2[b'labels'] = dataset[b'labels'][:500]
        self.small_dataset2[b'data'] = dataset[b'data'][:500]

    def test_same_label_results_in_close_histogram(self):
        nb_bins = 255

        imgs1 = extract_images_from_label(self.small_dataset1, 0)
        imgs2 = extract_images_from_label(self.small_dataset2, 0)
        hist1 = images_to_histogram(imgs1, histogram_from_grayscale, nb_bins)
        hist2 = images_to_histogram(imgs2, histogram_from_grayscale, nb_bins)

        result = compare_histogram(hist1, hist2)
        self.assertLess(result, 1)

    def test_different_label_results_in_diverse_histogram(self):
        nb_bins = 255

        imgs1 = extract_images_from_label(self.small_dataset1, 0)
        hist1 = images_to_histogram(imgs1, histogram_from_grayscale, nb_bins)
        for i in range(1, 9):
            imgs2 = extract_images_from_label(self.small_dataset2, i)
            hist2 = images_to_histogram(imgs2, histogram_from_grayscale, nb_bins)

            result = compare_histogram(hist1, hist2)
            self.assertGreater(result, 1)
        
    def test_histogram_from_contrast(self):
        nb_bins = 255
        imgs1 = extract_images_from_label(self.small_dataset1, 0)
        imgs2 = extract_images_from_label(self.small_dataset2, 0)
        hist1 = images_to_histogram(imgs1, histogram_from_contrast, nb_bins)
        hist2 = images_to_histogram(imgs2, histogram_from_contrast, nb_bins)
        original = compare_histogram(hist1, hist2)

        for i in range(1, 9):
            imgs2 = extract_images_from_label(self.small_dataset2, i)
            hist2 = images_to_histogram(imgs2, histogram_from_contrast, nb_bins)

            result = compare_histogram(hist1, hist2)
            self.assertGreater(result, original)

if __name__ == '__main__':
    unittest.main()