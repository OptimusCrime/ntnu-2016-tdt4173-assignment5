import logging
import numpy as np

from PIL import Image
from PIL import ImageDraw
from skimage.transform import resize
from skimage.transform import pyramid_gaussian
from skimage import io

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
                    log.info('Preprocessing data with %s technique' % repr(processing))
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
        # result = model.predict(X_test)
        result = None

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

            # Define window size
            (win_width, win_height) = (20, 20)
            # Loop the images and get fragments
            for i in range(len(images)):
                # Shortcut images
                image = images[i]

                # Get the fragments
                log.info('Get fragments from image number %i.' % (i + 1))
                for (index, scaled_image) in self.image_pyramid_down(image):
                    print('Shape of scaled image', scaled_image.shape)
                    for (x, y, window) in self.sliding_window(scaled_image, step_size=4, window_size=(win_width, win_height)):
                        if window.shape[0] != win_height or window.shape[1] != win_width:
                            continue
                        # HERE WE EXECUTE PREDICT ON WINDOW
                        import copy
                        from skimage import io

                        result = model.predict_proba([window.reshape((400, ))])
                        max_value = result[0].max()
                        if max_value >= 0.8:
                            letter_index = np.argmax(result[0])
                            print(chr(letter_index + 97))
                            io.imshow(window)
                            io.show()
                        # fragments, locations = es.extract(scaled_image)
                        # log.info('Got %i fragments from image number %i, with scale %s' % (len(fragments), (i + 1), scale))
                exit()

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
                        print("Location = " + str(locations[j]))

                        dr.rectangle(((locations[j][0], locations[j][1]), (
                                locations[j][0] + 20, locations[j][1] + 20)), outline="blue")


                # Get save path
                save_path_split = paths[i].split('.')
                save_path_split_clean = save_path_split[0] + '_' + str(scale) + '_output.' + save_path_split[1]

                im.save(save_path_split_clean, "JPEG")

                '''
                image_data.append({
                    'file_name': 'foobar',
                    'original': image,
                    'fragments': es.extract(image)
                })'''

            #print(image_data)

    @staticmethod
    def image_pyramid_down(image, downscale=1.5, min_size=(400, 400)):
        for (i, resized) in enumerate(pyramid_gaussian(image, downscale=downscale)):
            if resized.shape[0] < min_size[1] or resized.shape[1] < min_size[0]:
                break

            yield i, resized

    @staticmethod
    def image_pyramid_up(image, upscale=1.5, max_size=(1200, 1200)):
        pass

    @staticmethod
    def sliding_window(image, step_size, window_size):
        print(image.shape)
        for y in range(0, image.shape[0], step_size):
            for x in range(0, image.shape[1], step_size):
                yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])
