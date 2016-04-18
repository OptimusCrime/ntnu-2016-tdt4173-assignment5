import abc


class BaseLoader(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def load(self):
        """
        Method that should initiate loading and return data set
        :return Data set
        """

    @abc.abstractmethod
    def preprocess(self):
        """
        Method that should preprocess the dataset
        :return: Preprocessed dataset
        """