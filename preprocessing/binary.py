import numpy as np
from .base import BasePreprocessing


class BinaryProcessing(BasePreprocessing):

    def process(self, dataset):
        if dataset is None:
            return None

        # Loop each image in the dataset
        for i in range(len(dataset)):

            old_shape = None
            if len(dataset[i].shape) > 1:
                old_shape = dataset[i].shape
                dataset[i] = dataset[i].reshape(((old_shape[0] * old_shape[1]), ))


            # Find the average color in the dataset
            average_value = np.average(dataset[i])

            # If the average value is over 0.5 we have more black than white and we should negate the values
            negate = False
            if average_value >= 0.5:
                negate = True

            # Loop each pixel value in the image
            for j in range(len(dataset[i])):
                # Binary
                if negate:
                    dataset[i][j] = 1 if dataset[i][j] <= 0.50 else 0
                else:
                    dataset[i][j] = 0 if dataset[i][j] <= 0.50 else 1

            if old_shape is not None:
                dataset[i] = dataset[i].reshape((old_shape[0], old_shape[1]))

        # Return the final list of images
        return dataset

    def __repr__(self):
        return 'BinaryProcessing'
