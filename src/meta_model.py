#!/usr/bin/env python

import os

from src.logger import HuliLogging
from src.utils import MODELS_DIR

logger = HuliLogging.get_logger(__name__)


EXTENSION_H5 = '.h5'

UNKNOWN_EPOCH = -1


class MetaModel:

    def __init__(self, name, keras_model=None, tflite_interpretter=None, epoch=UNKNOWN_EPOCH):
        self.name = name
        self.keras_model = keras_model
        self.epoch = epoch
        self.tflite_interpretter = tflite_interpretter

    def compile(self, *args, **kwargs):
        assert self.keras_model is not None
        return self.keras_model.compile(*args, **kwargs)

    def fit(self, *args, epochs=1, **kwargs):
        assert self.keras_model is not None
        result = self.keras_model.fit(*args, epochs=epochs, **kwargs)
        self.epoch = epochs
        return result

    def evaluate(self, *args, **kwargs):
        assert self.keras_model is not None
        return self.keras_model.evaluate(*args, **kwargs)

    def save(self, **kwargs):
        assert self.keras_model is not None
        filepath = os.path.join(MODELS_DIR, self.filename_h5())
        logger.info('Saving %s...', filepath)
        return self.keras_model.save(filepath, **kwargs)

    def summary(self, *args, **kwargs):
        assert self.keras_model is not None
        return self.keras_model.summary(*args, **kwargs)

    def filename_no_ext(self):
        return '%s_%03d' % (self.name, self.epoch)

    def filename_h5(self):
        return '%s%s' % (self.filename_no_ext(), EXTENSION_H5)
