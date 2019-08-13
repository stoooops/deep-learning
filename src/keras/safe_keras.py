#!/usr/bin/env python
#
# Wrapper around tensorflow.keras static functions, to return exceptions instead as error codes
#
from .errors import ERROR_KERAS_CAUGHT_EXCEPTION
from src.utils.logger import Logging

from tensorflow import keras

logger = Logging.get_logger(__name__)


def load_model(filepath):
    assert filepath is not None and isinstance(filepath, str)
    try:
        keras_model = keras.models.load_model(filepath)
    except IOError as e:
        logger.exception(e)
        return ERROR_KERAS_CAUGHT_EXCEPTION, None
    except Exception as e:
        logger.exception(e)
        return ERROR_KERAS_CAUGHT_EXCEPTION, None

    return 0, keras_model
