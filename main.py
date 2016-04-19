import logging
from logging.config import dictConfig
from config.configuration import LOG_CONFIG

from load.loader import Chars74KLoader
from load.image_load import ImageLoader
from models.models import k_nearest_neighbours as knn
from models.models import support_vector_machine as svm

from preprocessing.binary import BinaryPreprocessing
from preprocessing.normalize import NormalizePreprocessing
from preprocessing.hog import DenoisePreprocessing

from ocr import OCR

if __name__ == '__main__':
    # Setup logging
    dictConfig(LOG_CONFIG)
    log = logging.getLogger(__name__)

    training_loader = Chars74KLoader(config={
        'from_pickle': True,
        'percent_to_train_data': 0.8
    })

    # Define the model
    model = knn(from_pickle=False)
    # model = svm(from_pickle=False)

    # Define the preprocessing
    pre_processing = [BinaryPreprocessing()]

    # Loader for images/recognizer
    image_data_loader = ImageLoader(config={})

    # Call the OCR
    ocr = OCR(model=model,
              training_data_loader=training_loader,
              preprocessing=pre_processing,
              image_data_loader=image_data_loader)