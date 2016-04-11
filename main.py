import logging
from logging.config import dictConfig
from config.configuration import LOG_CONFIG

from load.loader import Chars74KLoader
from ocr import OCR

if __name__ == '__main__':
    # Setup logging
    dictConfig(LOG_CONFIG)
    log = logging.getLogger(__name__)

    loader = Chars74KLoader({
        'from_pickle': True
    })

    ocr = OCR(loader)
