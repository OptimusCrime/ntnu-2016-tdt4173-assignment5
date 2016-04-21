import logging
import time
import glob
import os
import numpy as np

from PIL import Image
from PIL import ImageDraw
from datetime import datetime
from load.pickling import pickle_data, unpickle_data
from load.chars74k_load import Chars74KLoader
from skimage.transform import pyramid_gaussian
from sklearn.metrics import classification_report, confusion_matrix

from load.base import BaseLoader
from preprocessing.base import BasePreprocessing

np.set_printoptions(linewidth=200)
__pickled_data_directory__ = os.path.join('.', 'data', 'pickled-classifier')

OCR_BASE_CONFIG = {
    'window_size': (20, 20),
    'sliding_window_step_size': 4,
    'prediction_threshold': 0.9,
    'window_content': 90,
    'do_initial_prediction': True,
    'pre_processing': None,
    'data_from_pickle': False,
    'model_from_pickle': False
}


class OCR(object):

    def __init__(self, training_data_loader=None, image_data_loader=None,  model=None, config=OCR_BASE_CONFIG):
        self._log = logging.getLogger(__name__)
        self._log.info('Initiating OCR')

        self.config = OCR_BASE_CONFIG
        self.config.update(config)

        # Make sure we have a loader
        if not isinstance(training_data_loader, BaseLoader):
            if not self.config['data_from_pickle']:
                raise ValueError('When data_from_pickle is false; "training_data_loader" must be of type BaseLoader')

        self.pre_processing = self.config['pre_processing']

        if not self.config['data_from_pickle']:
            self.training_data_loader = training_data_loader
            X_train, y_train, X_test, y_test = self.training_data_loader.load()
            # Check if any pre-processing was supplied. Run if supplied.
            if isinstance(self.pre_processing, list):
                for processing in self.pre_processing:
                    if isinstance(processing, BasePreprocessing):
                        self._log.info('Pre-processing data set with %s technique' % repr(processing))
                        X_train = processing.process(X_train)
                        X_test = processing.process(X_test)
            # Save data set to pickled file
            self._log.info('Saving pre-processed data set to pickled file')
            Chars74KLoader.save_data_set_to_pickle((X_train, y_train, X_test, y_test))
            self._log.info('Saved pre-processed data set to pickled fie')
        else:
            self._log.info('Loading data set from pickled file')
            try:
                X_train, y_train, X_test, y_test = Chars74KLoader.load_data_set_from_pickle()
            except TypeError:
                raise TypeError('no data set found from pickled files and data set cannot be None')

        # If model should not be loaded from pickled file
        if not self.config['model_from_pickle']:
            self.model = model
            self._log.info('Starting fitting of model')
            self.model.fit(X_train, y_train)
            self._log.info('Done fitting model')
            self._log.info('Saving model ')
            self.save_classifier_to_pickle(self.model)
            self._log.info('Saved model')
        else:
            self._log.info('Loading classifier from pickled file')
            self.model = self.load_classifier_from_pickle()
            # If no model found from pickled models
            if not self.model:
                raise ValueError('no classifier found from pickled files and model cannot be None')

        # To a prediction to get correctness of model
        if self.config['do_initial_prediction']:
            self._log.info('Predicting test set of size: %i' % len(X_test))
            result = self.model.predict(X_test)
            self._log.info('Classification report %s\n%s' % (self.model, classification_report(y_test, result)))
            self._log.info('Confusion matrix:\n%s' % confusion_matrix(y_test, result))
            # if result is not None:
            #     self._log.info('%.2f percent correct' % (sum([1 if result[i] == y_test[i] else 0 for i in range(len(result))]) / len(result) * 100))

        # Try to recognize letters on images in recognize
        if image_data_loader is not None and isinstance(image_data_loader, BaseLoader):
            # Load the images
            images, paths = image_data_loader.load()

            if isinstance(self.pre_processing, list):
                for processing in self.pre_processing:
                    if isinstance(processing, BasePreprocessing):
                        self._log.info('Pre-processing images with %s technique' % repr(processing))
                        images = processing.process(images)

            # Loop the images and get fragments
            for i in range(len(images)):
                # Shortcut images
                image = images[i]
                classifications = []

                im = Image.open(paths[i])
                dr = ImageDraw.Draw(im)

                # Get the fragments / sub-windows
                self._log.info('Generating fragments from image (%i): %s' % (i + 1, paths[i].split('/')[-1]))
                for (y, x, window) in self.sliding_window(image, step_size=self.config['sliding_window_step_size'], window_size=self.config['window_size']):
                    if window.shape[0] != self.config['window_size'][1] or window.shape[1] != self.config['window_size'][0]:
                        continue

                    # Here we execute the prediction
                    if np.count_nonzero(window) > 0 and np.count_nonzero(window) >= self.config['window_content']:
                        result = self.model.predict_proba([window.reshape((400, ))])
                        max_value = result[0].max()
                        if max_value >= self.config['prediction_threshold']:
                            letter_index = np.argmax(result)
                            letter = chr(letter_index + 97)  # Get the predicted character
                            classifications.append((letter, result[0]))  # Append tuple of character and probability
                            dr.rectangle(((x, y), (x + 20, y + 20)), outline="blue")

                # Get save path
                save_path_split = paths[i].split('.')
                save_path_split_clean = save_path_split[0] + '_output.' + save_path_split[1]
                im.save(save_path_split_clean, "JPEG")
                self._log.info('Got the following classifications (%i): %s' % (len(classifications), [i[0] for i in classifications]))

    def sliding_window(self, image, step_size, window_size):
        """
        Method to return fragments of image, given window_size and step_size
        :param image: The image to perform sliding on
        :param step_size: How large incrementation one iteration should be
        :param window_size: The wanted window size
        :return: Generator with windows/fragments
        """
        self._log.debug('Sliding image of size: %s with step: %s' % (str(image.shape), str(step_size)))
        for y in range(0, image.shape[0], step_size):
            for x in range(0, image.shape[1], step_size):
                yield (y, x, image[y:(y + window_size[1]), x:(x + window_size[0])])

    def save_classifier_to_pickle(self, pay_load, file_name=None):
        """
        Saves data to a GZipped binary dump of the data set
        :param file_name:
        :param pay_load:
        """
        if not file_name:
            file_name = '%s.%f.%s.classifier.gz' % (datetime.now().strftime('%Y-%m-%d'), time.clock(), pay_load.__class__.__name__)
            file_name = os.path.join(__pickled_data_directory__, file_name)

        pickle_data(pay_load, file_name)
        self._log.debug('Saved classifier to file: %s' % file_name)

    def load_classifier_from_pickle(self, file_name=None):
        """
        Loads data from a GZipped binary dump of the data set
        Takes thw newest if no file name is provided
        :param file_name:
        """
        if not file_name:
            try:
                file_name = max(glob.glob(os.path.join(__pickled_data_directory__, '*.classifier.gz')), key=os.path.getctime)
            except ValueError as e:
                self._log.error('Unable to load classifier from file since no pickled files could be found, ')
                return None

        self._log.debug('Loading classifier from file: %s' % file_name)
        return unpickle_data(file_name)

    @staticmethod
    def image_pyramid_down(image, downscale=1.5, min_size=(30, 30)):
        """
        Method to downscale images and yield scaled images down to the provided min_size
        :param image: Image to downscale
        :param downscale: Downscale factor
        :param min_size: Minimum image size
        :return: Generator with scaled images
        """
        for i, resized_image in enumerate(pyramid_gaussian(image, downscale=downscale)):
            if resized_image.shape[0] < min_size[1] or resized_image.shape[1] < min_size[0]:
                break

            yield i, resized_image
