import logging
import os
import numpy as np
import glob
import time

from skimage import io
from datetime import datetime
from math import floor

from load.pickling import pickle_data, unpickle_data
from load.base import BaseLoader

log = logging.getLogger(__name__)

__pickled_data_directory__ = os.path.join('.', 'data', 'pickled')

CHARS74KLOADER_BASE_CONFIG = {
    'percent_to_train_data': 0.9,
    'img_size': (20, 20),
    'images': 7112,
    'normalize': True,
    'from_pickle': False
}


class Chars74KLoader(BaseLoader):
    """
    Class for loading Chars74K data set
    TODO: Add functionality for data augmentation
    TODO: The data set should maybe return train and test data?
    """
    def __init__(self, config=CHARS74KLOADER_BASE_CONFIG):
        self.root_directory = os.path.abspath('data/chars74k-lite')
        self.config = CHARS74KLOADER_BASE_CONFIG
        self.config.update(config)

    def load(self):
        log.info('Initiating load of Chars74K data set')
        # If from_pickle, then load the data from pickled file
        # Create pickled file otherwise
        if self.config['from_pickle']:
            data = self.load_data_set_from_pickle()
            if data:
                log.info('Loaded %i images of %s pixels from pickled file' % (len(data[0]) + len(data[2]), self.config['img_size']))
                return data  # Tuple of (NP_Array(Vectors) / Array(Labels))
            else:
                log.error('Unable to load pickled file, getting default data instead')

        image_paths, image_labels = self.get_all_image_paths()
        # data_image_vectors = np.zeros((len(image_paths),) + self.config['img_size'])  # Use if 2D array is wanted
        image_vectors = np.zeros((len(image_paths),) + (self.config['img_size'][0] ** 2,))  # 7112 * 20^2

        for index in range(len(image_paths)):
            raw_image = io.imread(image_paths[index], as_grey=True)  # As grey to get 2D without RGB
            if self.config['normalize']:
                raw_image = raw_image / 255.0  # Normalize features by dividing
            image_vectors[index] = raw_image.reshape(self.config['img_size'][0] ** 2)  # Reshape to 1D vector of length 20^2

        # Split data set into (X_train, y_train, X_test and y_test)
        dataset_tuple = self.split_data_set(image_vectors, image_labels)

        self.save_data_set_to_pickle(dataset_tuple)
        log.info('Loaded %i images of %s pixels' % (len(image_labels), self.config['img_size']))
        return dataset_tuple

    def preprocess(self):
        # TODO: Implement this
        pass

    def split_data_set(self, vectors, labels):
        """
        Split data set into (X_train, y_train, X_test and y_test) using the
        percent_as_test_data value.
        Return it as a tuple (X_train, y_train, X_test and y_test)
        :param vectors:
        :param labels:
        :return:
        """
        num_of_images = len(vectors)
        num_of_train_data = floor(self.config['percent_to_train_data'] * num_of_images)

        indices = np.random.permutation(num_of_images)  # A random permutation of all indices
        X_train = [vectors[i] for i in indices[:num_of_train_data]]
        y_train = [labels[i] for i in indices[:num_of_train_data]]
        X_test = [vectors[i] for i in indices[num_of_train_data:]]
        y_test = [labels[i] for i in indices[num_of_train_data:]]

        return X_train, y_train, X_test, y_test

    def get_all_image_paths(self):
        """
        Gets and returns all paths for Chars74K images
        :return: The image paths and labels
        """
        image_paths, image_labels = [], []
        for directory_name, subdirectory_list, file_list in os.walk(self.root_directory):
            for file_name in file_list:
                if file_name.endswith(('.jpg',)):
                    image_paths.append(os.path.join(directory_name, file_name))
                    # Translates labels to 0-26 as recommended in the exercise description
                    image_labels.append(ord(directory_name[-1]) - 97)
        return image_paths, image_labels

    @staticmethod
    def save_data_set_to_pickle(payload, filename=None):
        """
        Saves data to a GZipped binary dump of the data set
        """
        if not filename:
            filename = '%s%f.chars74k-lite.gz' % (datetime.now().strftime('%Y-%m-%d'), time.clock())
            filename = os.path.join(__pickled_data_directory__, filename)

        pickle_data(payload, filename)
        log.debug('Saved data set to file: %s' % filename)


    @staticmethod
    def load_data_set_from_pickle(filename=None):
        """
        Loads data from a GZipped binary dump of the data set
        Takes thw newest if no file name is provided
        """
        if not filename:
            try:
                filename = max(glob.glob(os.path.join(__pickled_data_directory__, '*.chars74k-lite.gz')), key=os.path.getctime)
            except ValueError as e:
                log.error('Unable to load data set from file since no pickled files could be found, ')
                return None

        log.debug('Loading data set from file: %s' % filename)
        return unpickle_data(filename)
