import pickle
import gzip
import logging

log = logging.getLogger(__name__)


def pickle_data(payload, filename):
    """
    Saves payload to binary gZipped file
    :param payload: Payload to save
    :param filename: Filename to save to (as full path)
    """
    if filename is None:
        raise ValueError('filename must be defined (as full path)')
    if payload is None:
        raise ValueError('payload must be provided')

    try:
        with gzip.open(filename, 'wb') as f:
            pickle.dump(payload, f)
    except OSError as e:
        log.error('Could not open file: %s (%s)' % (filename, e))


def unpickle_data(filename):
    """
    Loads payload from binary file, and returns it
    :param filename: Filename to load (as full path)
    :return: The loaded binary file
    """
    if not filename:
        raise ValueError('filename must be defined (as full path)')

    try:
        with gzip.open(filename) as f:
            return pickle.load(f)
    except OSError as e:
        log.error('Could not open file: %s (%s)' % (filename, e))
        return None
