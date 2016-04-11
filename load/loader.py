import abc
import logging
import os
import numpy as np
import pickle
import gzip
import glob
import time

from skimage import io
from datetime import datetime

log = logging.getLogger(__name__)

__data_directory__ = os.path.join('.', 'data', 'pickled')

CHARS_BASE_CONFIG = {
    'img_size': (20, 20),
    'images': 7112,
    'normalize': True,
    'from_pickle': False
}


class BaseLoader(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def load(self):
        """
        Method that should initiate loading and return data set
        :return Data set
        """


class ImageLoader(BaseLoader):
    """
    Class for laoding image files
    """

    def __init__(self):
        pass

    def load(self):
        pass


class Chars74KLoader(BaseLoader):
    """
    Class for loading Chars74K data set
    """
    def __init__(self, config=CHARS_BASE_CONFIG):
        self.root_directory = os.path.abspath('data/chars74k-lite')
        self.config = CHARS_BASE_CONFIG
        self.config.update(config)

    def load(self):
        log.info('Initiating load of Chars74K data set')
        if self.config['from_pickle']:
            data = self.load_from_pickle()
            log.info('Loaded %i images of %s pixels from pickled file' % (len(data[0]), self.config['img_size']))
            return data  # Tuple of (NP_Array(Vectors) / Array(Labels))
        else:
            image_paths, image_labels = self.get_all_image_paths()
            data_image_vectors = np.zeros((len(image_paths),) + self.config['img_size'])

            for index in range(len(image_paths)):
                raw_image = io.imread(image_paths[index], as_grey=True)
                if self.config['normalize']:
                    raw_image = raw_image / 255.0  # Normalize features
                data_image_vectors[index] = raw_image

            self.save_to_pickle((data_image_vectors, image_labels))
            log.info('Loaded %i images of %s pixels' % (len(image_labels), self.config['img_size']))
            return data_image_vectors, image_labels

    def get_all_image_paths(self):
        """
        Gets and returns all possible paths for Chars74K images
        :return: The image paths and labels
        """
        image_paths, image_labels = [], []
        for directory_name, subdirectory_list, file_list in os.walk(self.root_directory):
            for file_name in file_list:
                if file_name.endswith(('.jpg',)):
                    image_paths.append(os.path.join(directory_name, file_name))
                    image_labels.append(directory_name[-1])
        return image_paths, image_labels

    @staticmethod
    def save_to_pickle(payload, file_name=None):
        """
        Saves data to a GZipped binary dump of the data set
        """
        if not file_name:
            file_name = '%s%f.chars74k-lite.gz' % (datetime.now().strftime('%Y-%m-%d'), time.clock())
            file_name = os.path.join(__data_directory__, file_name)

        try:
            with gzip.open(file_name, 'wb') as f:
                pickle.dump(payload, f)
        except OSError as e:
            log.error('Could not open file: %s (%s)' % (file_name, e))
        log.debug('Saved file: %s' % file_name)


    @staticmethod
    def load_from_pickle(file_name=None):
        """
        Loads data from a GZipped binary dump of the data set
        Takes thw newest if no file name is provided
        """
        if not file_name:
            file_name = max(glob.glob(os.path.join(__data_directory__, '*.chars74k-lite.gz')), key=os.path.getctime)

        log.debug('Loading file: %s' % file_name)
        try:
            with gzip.open(file_name) as f:
                return pickle.load(f)
        except OSError as e:
            log.error('Could not open file: %s (%s)' % (file_name, e))








