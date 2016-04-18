import numpy as np

from .base import BasePreprocessing


class NormalizePreprocessing(BasePreprocessing):

    def __init__(self, normalize_value=255.0):
        self.normalize_value = normalize_value

    def process(self, dataset):
        if dataset is None:
            return None

        normalized_images = []
        # Iterate every image vector and divide by normalize value
        for image in dataset:
            normalized_images.append(image / self.normalize_value)

        return np.array(normalized_images)
