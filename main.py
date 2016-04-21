import logging
from logging.config import dictConfig
from config.configuration import LOG_CONFIG

from load.chars74k_load import Chars74KLoader
from load.image_load import ImageLoader

from models.models import k_nearest_neighbours as knn
from models.models import support_vector_machine as svm
from models.models import decision_tree as tree
from preprocessing.binary import BinaryProcessing
from preprocessing.normalize import NormalizeProcessing
from preprocessing.denoise import DenoiseProcessing
from ocr import OCR

if __name__ == '__main__':
    # Setup logging
    dictConfig(LOG_CONFIG)
    log = logging.getLogger(__name__)

    # Loader for images/recognizer
    image_data_loader = ImageLoader()

    # Loader for training set
    training_loader = Chars74KLoader(config={
        'percent_to_train_data': 0.8,
        'extend_data_set': True,
        'noise_types': ('gaussian', 's&p')
    })

    ocr_configuration = {
        'pre_processing': [BinaryProcessing()],
        'do_initial_prediction': False,
        'prediction_threshold': 0.85,
        'window_content': 80,
        'data_from_pickle': True,
        'model_from_pickle': True
    }

    # Define the model
    # model = None
    # model = svm()
    # model = tree()
    model = knn()

    # Call the OCR
    ocr = OCR(model=model,
              training_data_loader=training_loader,
              image_data_loader=image_data_loader,
              config=ocr_configuration)
