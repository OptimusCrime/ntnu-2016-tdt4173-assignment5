import logging

from load.base import BaseLoader

log = logging.getLogger(__name__)


class OCR(object):

    def __init__(self, model=None, data_loader=None):
        if not isinstance(data_loader, BaseLoader):
            raise ValueError('data_loader must be of type BaseLoader')
        # TODO: Same as above for model?
        log.info('Initiating OCR')
        self.data_loader = data_loader  # Must be of type BaseLoader
        self.model = model
        training_data = self.data_loader.load()  # Returned as (X_train, y_train, X_test, y_test)

        """
        USE FOR SHOWING A SINGLE IMAGE (RESHAPES TO 2D)
        io.imageshow(np.reshape(training_data[0][0], (20, 20)))
        io.show()
        """

        # Test with kNN
        model.fit(training_data[0], training_data[1])
        result = model.predict(training_data[2])

        print('%.2f percent correct' % (sum([1 if result[i] == training_data[3][i] else 0 for i in range(len(result))]) / len(result) * 100))
