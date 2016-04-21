import logging

from load.pickling import pickle_data, unpickle_data
from sklearn import neighbors
from sklearn import svm
from sklearn import tree

log = logging.getLogger(__name__)


def k_nearest_neighbours(from_pickle=False):
    log.info('Initiating kNN model')
    return neighbors.KNeighborsClassifier(n_neighbors=5)


def support_vector_machine(from_pickle=False):
    log.info('Initiating SVM model')
    return svm.SVC(gamma=0.005, probability=True)


def decision_tree():
    log.info('Initiating DecisionTree model')
    return tree.DecisionTreeClassifier(max_depth=20)
