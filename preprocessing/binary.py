
import numpy as np
from .base import BasePreprocessing

class BinaryPreprocessing(BasePreprocessing):

    def process(self, dataset):
        images = [0, 2]

        # Loop the sets
        for i in images:

            # Loop each image in the dataset
            for j in range(len(dataset[i])):
                # Find the average color in the dataset
                average_value = np.average(dataset[i][j])

                # If the average value is over 0.5 we have more black than white and we should negate the values
                negate = False
                if average_value >= 0.5:
                    negate = True

                # Loop each pixel value in the image
                for k in range(len(dataset[i][j])):
                    # Binary
                    if negate:
                        dataset[i][j][k] = 1 if dataset[i][j][k] <= 0.5 else 0
                    else:
                        dataset[i][j][k] = 0 if dataset[i][j][k] <= 0.5 else 1

        return dataset