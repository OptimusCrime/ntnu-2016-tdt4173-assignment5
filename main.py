import logging
from logging.config import dictConfig
from config.configuration import LOG_CONFIG

if __name__ == '__main__':
    # Setup logging
    dictConfig(LOG_CONFIG)
    log = logging.getLogger(__name__)