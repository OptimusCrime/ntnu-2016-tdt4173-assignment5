
import numpy as np
from .base import BasePreprocessing


class BinaryPreprocessing(BasePreprocessing):

    def process(self, dataset, shape=None):
        if dataset is None:
            return None

        # Loop each image in the dataset
        output = []
        for i in range(len(dataset)):
            image = None

            reshape = False
            shape = None
            if len(dataset[i].shape) > 1:
                reshape = True
                shape = dataset[i].shape

                image = dataset[i].reshape(((shape[0] * shape[1]),))
            else:
                image = dataset[i]

            # Find the average color in the dataset
            average_value = np.average(image)

            # If the average value is over 0.5 we have more black than white and we should negate the values
            negate = False
            if average_value >= 0.5:
                negate = True

            # Loop each pixel value in the image
            for j in range(len(image)):
                # Binary
                if negate:
                    image[j] = 1 if image[j] <= 0.5 else 0
                else:
                    image[j] = 0 if image[j] <= 0.5 else 1

            if reshape:
                output.append(image.reshape(shape))
            else:
                output.append(image)

        return output

    def __repr__(self):
        return 'BinaryProcessing'
