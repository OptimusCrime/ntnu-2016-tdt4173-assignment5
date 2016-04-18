import numpy as np

from .base import BasePreprocessing

from skimage import feature, exposure
from skimage.restoration import denoise_tv_chambolle
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square

from skimage import io


class HogPreprocessing(BasePreprocessing):

    def process(self, dataset):
        if dataset is None:
            return None

        processed_images = []

        for image in range(len(dataset)):
            img = dataset[image]
            # fd, img = feature.hog(np.reshape(image, (20, 20)), orientations=8, pixels_per_cell=(20, 20), cells_per_block=(1, 1), visualise=True)
            # hog_image_rescaled = exposure.rescale_intensity(img, in_range=(0, 0.02))
            # io.imsave('out.jpg', hog_image_rescaled)
            # print(fd)
            # print(img)
            img = denoise_tv_chambolle(img, weight=0.1)
            # print(edge_sobel)
            # io.imshow(np.reshape(image, (20, 20)))
            # io.show()
            # io.imshow(edge_sobel)
            # io.show()
            # io.imshow(fd)
            # io.show()
            processed_images.append(img)

        return np.array(processed_images)

    def __repr__(self):
        return 'HogProcessing'
