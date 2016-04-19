import numpy as np

from .base import BasePreprocessing

from skimage.restoration import denoise_tv_chambolle


class DenoisePreprocessing(BasePreprocessing):

    def process(self, dataset):
        if dataset is None:
            return None

        processed_images = []

        for image in range(len(dataset)):
            img = dataset[image]
            img = denoise_tv_chambolle(img, weight=0.1)
            processed_images.append(img)

        return np.array(processed_images)

    def __repr__(self):
        return 'DenoiseProcessing'
