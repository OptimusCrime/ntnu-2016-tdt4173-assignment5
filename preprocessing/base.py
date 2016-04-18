import abc


class BasePreprocessing(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def process(self, dataset):
        pass
