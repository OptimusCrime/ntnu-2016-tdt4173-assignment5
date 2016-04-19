import numpy as np

from .base import BasePreprocessing

from skimage.restoration import denoise_tv_chambolle


class DenoisePreprocessing(BasePreprocessing):

    def process(self, dataset):
        if dataset is None:
            return None

        # Loop each image in the dataset
        for i in range(len(dataset)):
            # Run the denoise method
            old_shape = None
            if len(dataset[i].shape) == 1:
                old_shape = dataset[i].shape
                dataset[i] = denoise_tv_chambolle(dataset[i].reshape((20, 20)), weight=0.1)
            else:
                dataset[i] = denoise_tv_chambolle(dataset[i], weight=0.1)

            # Reshape image to 1d 400 array
            if old_shape is not None:
                dataset[i] = dataset[i].reshape((400, ))

        # Return the dataset
        return dataset

    def __repr__(self):
        return 'DenoiseProcessing'
