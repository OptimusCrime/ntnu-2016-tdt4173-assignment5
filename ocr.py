import logging

from load.base import BaseLoader
from preprocessing.base import BasePreprocessing

log = logging.getLogger(__name__)


class OCR(object):

    def __init__(self, model=None, training_data_loader=None, preprocessing=None, image_data_loader=None):
        # Make sure we have a loader
        if not isinstance(training_data_loader, BaseLoader):
            raise ValueError('training_data_loader must be of type BaseLoader')

        # Logging
        log.info('Initiating OCR')

        # Set some variables
        self.training_data_loader = training_data_loader
        self.model = model

        # Load the training data. Returned as (X_train, y_train, X_test, y_test)
        X_train, y_train, X_test, y_test = self.training_data_loader.load()

        # Check if any preprocessing was supplied. Run if supplied.
        if isinstance(preprocessing, tuple):
            for processing in preprocessing:
                if isinstance(processing, BasePreprocessing):
                    log.info('Preprocessing data with %s technique' % type(processing))
                    X_train = processing.process(X_train)
                    X_test = processing.process(X_test)


        """
        USE FOR SHOWING A SINGLE IMAGE (RESHAPES TO 2D)
        io.imageshow(np.reshape(training_data[0][0], (20, 20)))
        io.show()
        """

        model.fit(X_train, y_train)
        result = model.predict(X_test)

        log.info('%.2f percent correct' % (sum([1 if result[i] == y_test[i] else 0 for i in range(len(result))]) / len(result) * 100))

        if result is not None:
            log.info('%.2f percent correct' % (sum([1 if result[i] == training_data[3][i] else 0 for i in range(len(result))]) / len(result) * 100))

        # Try to recognize letters on images in recognize
        if image_data_loader is not None and isinstance(image_data_loader, BaseLoader):
            image_data, image_shapes = image_data_loader.load()

            print(image_data)
            print(image_shapes)
