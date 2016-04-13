import logging
from logging.config import dictConfig
from config.configuration import LOG_CONFIG

from load.loader import Chars74KLoader
from models.knn import k_nearest_neighbours as knn
from ocr import OCR

if __name__ == '__main__':
    # Setup logging
    dictConfig(LOG_CONFIG)
    log = logging.getLogger(__name__)

    loader = Chars74KLoader(config={
        'from_pickle': True,
        'percent_to_train_data': 0.9
    })

    model = knn(from_pickle=False)

    ocr = OCR(model=model, data_loader=loader)