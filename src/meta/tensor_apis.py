#!/usr/bin/env python

import os

from abc import ABC, abstractmethod
from aenum import UniqueEnum

from src.utils.file_utils import MODELS_DIR

EXTENSION_H5 = '.h5'
EXTENSION_PB = '.pb'
EXTENSION_TFLITE = '_int8.tflite'


class TensorApi(UniqueEnum):
    NONE = 'none'
    TENSOR_FLOW = 'tensorflow'
    KERAS = 'keras'
    TF_LITE = 'tf.lite'
    TENSOR_RT = 'trt'


class AbstractTensorModel(ABC):

    def __init__(self, name):
        assert name is not None and isinstance(name, str)
        self.name = name

    @abstractmethod
    def compile(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def dump(self):
        pass

    def filename_no_ext(self, epoch):
        return '%s_%03d' % (self.name, epoch)

    def filename_h5(self, epoch):
        return '%s%s' % (self.filename_no_ext(epoch), EXTENSION_H5)

    def filename_pb(self, epoch):
        return '%s%s' % (self.filename_no_ext(epoch), EXTENSION_PB)

    def filename_tflite(self, epoch):
        return '%s%s' % (self.filename_no_ext(epoch), EXTENSION_TFLITE)

    def filepath_h5(self, epoch):
        return os.path.join(MODELS_DIR, self.filename_h5(epoch))

    def filepath_pb(self, epoch):
        return os.path.join(MODELS_DIR, self.filename_pb(epoch))

    def filepath_tflite(self, epoch):
        return os.path.join(MODELS_DIR, self.filename_tflite(epoch))