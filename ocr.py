import logging
import numpy as np

from skimage import io

from load.base import BaseLoader
from sklearn.neighbors import KNeighborsClassifier

log = logging.getLogger(__name__)


class OCR(object):

    def __init__(self, model=None, data_loader=None):
        if not isinstance(data_loader, BaseLoader):
            raise ValueError('data_loader must be of type BaseLoader')
        # TODO: Same as above for model?
        log.info('Initiating OCR')
        self.loader = data_loader  # Must be of type BaseLoader
        self.model = model
        training_data = self.loader.load()  # Returned as (X_train, y_train, X_test, y_test)

        """
        USE FOR SHOWING A SINGLE IMAGE (RESHAPES TO 2D)
        io.imageshow(np.reshape(training_data[0][0], (20, 20)))
        io.show()
        """

        knn = KNeighborsClassifier()
        knn.fit(training_data[0], training_data[1])
        result = knn.predict(training_data[2])
        print(sum([1 if result[i] == training_data[3][i] else 0 for i in range(len(result))]) / len(result) * 100)
