import numpy as np
from .base import BasePreprocessing


class BinaryProcessing(BasePreprocessing):

    def process(self, dataset):
        if dataset is None:
            return None

        # Loop each image in the dataset
        for i in range(len(dataset)):
            # Store old shape
            old_shape = None
            if len(dataset[i].shape) > 1:
                old_shape = dataset[i].shape
                dataset[i] = dataset[i].reshape(((old_shape[0] * old_shape[1]), ))

            # Loop each pixel value in the image
            for j in range(len(dataset[i])):
                # Binary
                dataset[i][j] = 1 if dataset[i][j] <= 0.50 else 0

            if old_shape is not None:
                dataset[i] = dataset[i].reshape((old_shape[0], old_shape[1]))

        # Return the final list of images
        return dataset

    def __repr__(self):
        return 'BinaryProcessing'
