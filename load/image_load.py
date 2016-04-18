import logging
import os
from skimage import io

from load.base import BaseLoader

log = logging.getLogger(__name__)

IMAGELOADER_BASE_CONFIG = {
    'normalize': True,
    'from_pickle': False
}


class ImageLoader(BaseLoader):
    """
    Class for loading Chars74K data set
    TODO: Add functionality for data augmentation
    TODO: The data set should maybe return train and test data?
    """
    def __init__(self, config=IMAGELOADER_BASE_CONFIG):
        self.root_directory = os.path.abspath('data/images')
        self.config = IMAGELOADER_BASE_CONFIG
        self.config.update(config)

    def load(self):
        log.info('Initiating load of images to recognize')

        # Get all paths
        image_paths = self.get_files()

        # Get the actual images
        images = []
        shapes = []
        for index in range(len(image_paths)):
            raw_image = io.imread(image_paths[index], as_grey=True)  # As grey to get 2D without RGB
            if self.config['normalize']:
                raw_image = raw_image / 255.0  # Normalize features by dividing
            height = len(raw_image)
            width = len(raw_image[0])

            # Add new image and reshape
            images.append(raw_image.reshape(height * width))

            # Store the shape
            shapes.append((width, height))

        # Debug
        log.info('Loaded %i image(s) to recognize' % (len(images)))

        # Return the images and the shapes
        return images, shapes

    def get_files(self):
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