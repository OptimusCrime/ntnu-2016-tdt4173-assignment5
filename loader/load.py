import abc


class BaseLoader(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def load(self):
        """
        Method that should initiate loading and return data set
        :return Data set
        """


class Loader(BaseLoader):

    def __init__(self):
        pass

    def load(self):
        pass





