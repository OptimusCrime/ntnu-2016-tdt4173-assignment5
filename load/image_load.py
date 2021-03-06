import logging
import os
import numpy as np
from skimage import io
from load.base import BaseLoader

IMAGE_LOADER_BASE_CONFIG = {
    'normalize': True,
    'from_pickle': False
}


class ImageLoader(BaseLoader):
    """
    Class for loading Image
    """
    def __init__(self, config=IMAGE_LOADER_BASE_CONFIG):
        self.root_directory = os.path.abspath('data/images')
        self.config = IMAGE_LOADER_BASE_CONFIG
        self.config.update(config)
        self._log = logging.getLogger(__name__)

    def load(self):
        self._log.info('Initiating load of images to recognize')

        # Get all paths
        image_paths = self.get_files()

        # Get the actual images
        images = []
        for index in range(len(image_paths)):
            raw_image = io.imread(image_paths[index], as_grey=True)  # As grey to get 2D without RGB
            if raw_image.dtype == np.uint8:
                raw_image = raw_image / 255.0
            images.append(raw_image)
            # Add new image and reshape
        # Debug
        self._log.info('Loaded %i image(s) to recognize' % (len(images)))

        # Return the images and the shapes
        return images, image_paths

    def get_files(self):
        """
        Returns all possible images to perform recognition on
        :return: Paths for every image
        """
        image_paths = []
        for directory_name, subdirectory_list, file_list in os.walk(self.root_directory):
            for file_name in file_list:
                # Only valid images
                if file_name.endswith(('.jpg', '.png',)):
                    # Filter out output images
                    output_image = False
                    file_name_split = file_name.split('.')

                    if len(file_name_split) > 1:
                        file_name_clean = file_name_split[-2]
                    else:
                        file_name_clean = file_name_split[-1]

                    # Output files have the form name_output
                    if '_' in file_name_clean:
                        file_name_cleaner = file_name_clean.split('_')

                        if file_name_cleaner[-1] == 'output':
                            output_image = True

                    # Only add if we are not dealing with output image
                    if not output_image:
                        image_paths.append(os.path.join(directory_name, file_name))

        return image_paths
