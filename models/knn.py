import logging

from load.pickling import pickle_data, unpickle_data
from sklearn import neighbors

log = logging.getLogger(__name__)


def k_nearest_neighbours(from_pickle=False):
    log.info('Initiating kNN model')
    if from_pickle:
        pass
        # TODO: Return pickled model

    return neighbors.KNeighborsClassifier()

