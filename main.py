import logging
from logging.config import dictConfig
from config.configuration import LOG_CONFIG

from load.loader import Chars74KLoader

from models.models import k_nearest_neighbours as knn
from models.models import support_vector_machine as svm

from preprocessing.binary import BinaryPreprocessing
from preprocessing.normalize import NormalizePreprocessing

from ocr import OCR

if __name__ == '__main__':
    # Setup logging
    dictConfig(LOG_CONFIG)
    log = logging.getLogger(__name__)

    loader = Chars74KLoader(config={
        'from_pickle': False,
        'percent_to_train_data': 0.9
    })

    # Define the model
    model = knn(from_pickle=False)
    # model = svm(from_pickle=False)

    # Define the preprocessing
    preprocessing = (NormalizePreprocessing(), BinaryPreprocessing())

    # Call the OCR
    ocr = OCR(model=model,
              data_loader=loader,
              preprocessing=preprocessing)