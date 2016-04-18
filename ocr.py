import logging

from extract_stuff import ExtractStuff
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

        if result is not None:
            log.info('%.2f percent correct' % (sum([1 if result[i] == y_test[i] else 0 for i in range(len(result))]) / len(result) * 100))

        # Try to recognize letters on images in recognize
        if image_data_loader is not None and isinstance(image_data_loader, BaseLoader):
            # Load the images
            images = image_data_loader.load()

            # Get all the pieces
            image_data = []

            es = ExtractStuff()

            # Loop the images and get fragments
            for image in images:
                image_data.append({
                    'file_name': 'foobar',
                    'original': image,
                    'fragments': es.extract(image)
                })

            print(image_data)
