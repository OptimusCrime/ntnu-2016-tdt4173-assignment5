import logging

log = logging.getLogger(__name__)


class OCR(object):

    def __init__(self, loader):
        log.info('Initiating OCR')
        self.loader = loader  # Must be of type BaseLoader
        training_data = self.loader.load()
        self.training_data_vectors = training_data[0]
        self.training_data_labels = training_data[1]
