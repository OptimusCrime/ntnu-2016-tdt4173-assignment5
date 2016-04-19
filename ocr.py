import logging
import numpy as np

from PIL import Image
from PIL import ImageDraw
from skimage.transform import pyramid_gaussian

from load.base import BaseLoader
from preprocessing.base import BasePreprocessing

log = logging.getLogger(__name__)

OCR_BASE_CONFIG = {
    'window_size': (20, 20),
    'sliding_window_step_size': 4,
    'prediction_threshold': 0.5,
    'window_content': 60
}


class OCR(object):

    def __init__(self, model=None, training_data_loader=None, preprocessing=None, image_data_loader=None, config=OCR_BASE_CONFIG):
        # Make sure we have a loader
        if not isinstance(training_data_loader, BaseLoader):
            raise ValueError('training_data_loader must be of type BaseLoader')

        log.info('Initiating OCR')

        self.config = OCR_BASE_CONFIG
        self.config.update(config)
        # Logging

        # Set some variables
        self.training_data_loader = training_data_loader
        self.model = model

        # Load the training data. Returned as (X_train, y_train, X_test, y_test)
        X_train, y_train, X_test, y_test = self.training_data_loader.load()

        # Check if any preprocessing was supplied. Run if supplied.
        if isinstance(preprocessing, list):
            for processing in preprocessing:
                if isinstance(processing, BasePreprocessing):
                    log.info('Pre-processing data with %s technique' % repr(processing))
                    X_train = processing.process(X_train)
                    X_test = processing.process(X_test)

        """
        USE FOR SHOWING A SINGLE IMAGE (RESHAPES TO 2D)
        io.imageshow(np.reshape(training_data[0][0], (20, 20)))
        io.show()
        """

        # Test with kNN
        log.info('Fitting model')
        model.fit(X_train, y_train)
        log.info('Predicting test set')
        result = model.predict(X_test)

        if result is not None:
            log.info('%.2f percent correct' % (sum([1 if result[i] == y_test[i] else 0 for i in range(len(result))]) / len(result) * 100))

        # Try to recognize letters on images in recognize
        if image_data_loader is not None and isinstance(image_data_loader, BaseLoader):
            # Load the images
            images, paths = image_data_loader.load()

            if isinstance(preprocessing, list):
                for processing in preprocessing:
                    if isinstance(processing, BasePreprocessing):
                        log.info('Pre-processing images with %s technique' % repr(processing))
                        images = processing.process(images)

            # Loop the images and get fragments
            for i in range(len(images)):
                # Shortcut images
                image = images[i]
                classifications = []

                im = Image.open(paths[i])
                dr = ImageDraw.Draw(im)

                # Get the fragments
                log.info('Get fragments from image number %i.' % (i + 1))
                for (y, x, window) in self.sliding_window(image, step_size=self.config['sliding_window_step_size'], window_size=self.config['window_size']):
                    if window.shape[0] != self.config['window_size'][1] or window.shape[1] != self.config['window_size'][0]:
                        continue

                    # HERE WE EXECUTE PREDICT ON WINDOW
                    if np.count_nonzero(window) > 0 and np.count_nonzero(window) >= self.config['window_content']:
                        result = model.predict_proba([window.reshape((400, ))])
                        max_value = result[0].max()
                        if max_value >= self.config['prediction_threshold']:
                            # print([window.reshape((400,))])
                            letter_index = np.argmax(result)
                            letter = chr(letter_index + 97)
                            classifications.append((letter, result[0]))
                            dr.rectangle(((x, y), (x + 20, y + 20)), outline="blue")

                # Get save path
                save_path_split = paths[i].split('.')
                save_path_split_clean = save_path_split[0] + '_output.' + save_path_split[1]
                im.save(save_path_split_clean, "JPEG")
                log.info('Got the following classifications (%i): %s' % (len(classifications), [i[0] for i in classifications]))


    @staticmethod
    def image_pyramid_down(image, downscale=1.5, min_size=(30, 30)):
        for (i, resized) in enumerate(pyramid_gaussian(image, downscale=downscale)):
            if resized.shape[0] < min_size[1] or resized.shape[1] < min_size[0]:
                break

            yield i, resized

    @staticmethod
    def image_pyramid_up(image, upscale=1.5, max_size=(1200, 1200)):
        pass

    @staticmethod
    def sliding_window(image, step_size, window_size):
        log.debug('Sliding image of size: %s with step: %s' % (str(image.shape), str(step_size)))
        for y in range(0, image.shape[0], step_size):
            for x in range(0, image.shape[1], step_size):
                yield (y, x, image[y:(y + window_size[1]), x:(x + window_size[0])])
