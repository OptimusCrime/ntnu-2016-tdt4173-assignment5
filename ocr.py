import logging
import numpy as np

from PIL import Image
from PIL import ImageDraw

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

        # Fit the model, aka training
        log.info('Begin train')
        model.fit(X_train, y_train)

        # Predict on the training set
        log.info('Begin predicting')
        result = model.predict(X_test)

        if result is not None:
            log.info('%.2f percent correct' % (sum([1 if result[i] == y_test[i] else 0 for i in range(len(result))]) / len(result) * 100))

        # Try to recognize letters on images in recognize
        if image_data_loader is not None and isinstance(image_data_loader, BaseLoader):
            # Load the images
            images, paths = image_data_loader.load()
            np.set_printoptions(threshold=np.inf)

            if isinstance(preprocessing, tuple):
                for processing in preprocessing:
                    if isinstance(processing, BasePreprocessing):
                        images = processing.process(images)

            # Get all the pieces
            image_data = []

            es = ExtractStuff()

            # Loop the images and get fragments
            for i in range(len(images)):
                # Shortcut images
                image = images[i]

                # Get the fragments
                log.info('Get fragments from image number %i.' % (i + 1))
                fragments, locations = es.extract(image)
                log.info('Got %i fragments from image number %i' % (len(fragments), (i + 1)))

                # Try to predict
                log.info('Predict on fragments')
                results = model.predict_proba(fragments)

                # Open image
                im = Image.open(paths[i])

                # Draw
                dr = ImageDraw.Draw(im)

                # Loop each of the results
                for j in range(len(results)):
                    # Find the maximum value
                    max_value = results[j].max()

                    # Check if the maximum value is above the threshold
                    if max_value >= 0.9:
                        # Get the letter
                        letter_index = np.argmax(results[j])

                        print("Letter = " + str(letter_index))
                        #print("Location = " + locations[j])

                        dr.rectangle(((locations[j][0], locations[j][1]), (
                                      locations[j][0] + 20, locations[j][1] + 20)), outline = "blue")


                # Get save path
                save_path_split = paths[i].split('.')
                save_path_split_clean = save_path_split[0] + '_output.' + save_path_split[1]

                im.save(save_path_split_clean, "JPEG")

                '''
                image_data.append({
                    'file_name': 'foobar',
                    'original': image,
                    'fragments': es.extract(image)
                })'''

            #print(image_data)
