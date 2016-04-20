import logging
from logging.config import dictConfig
from config.configuration import LOG_CONFIG

from load.chars74k_load import Chars74KLoader
from load.image_load import ImageLoader

from models.models import k_nearest_neighbours as knn
from models.models import support_vector_machine as svm
from preprocessing.binary import BinaryProcessing
from preprocessing.normalize import NormalizeProcessing
from preprocessing.denoise import DenoiseProcessing
from ocr import OCR

if __name__ == '__main__':
    # Setup logging
    dictConfig(LOG_CONFIG)
    log = logging.getLogger(__name__)

    # Loader for training set
    training_loader = Chars74KLoader(config={
        'from_pickle': True,
        'percent_to_train_data': 0.8,
        'extend_data_set': True,
        'noise_types': ('s&p', 'gaussian', 'poisson')
    })

    # Loader for images/recognizer
    image_data_loader = ImageLoader()

    # Define the model
    # model = svm()
    # model = knn()
    model = None

    # Define the pre-processing
    pre_processing = [BinaryProcessing()]

    # Call the OCR
    ocr = OCR(model=model,
              training_data_loader=training_loader,
              preprocessing=pre_processing,
              image_data_loader=image_data_loader,
              config={
                  'do_initial_prediction': True,
                  'prediction_threshold': 0.85,
                  'window_content': 80,
              })
