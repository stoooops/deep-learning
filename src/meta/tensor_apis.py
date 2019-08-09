#!/usr/bin/env python

import os

from abc import ABC, abstractmethod
from aenum import UniqueEnum

from src.meta.constants import *
from src.meta.metadata import Metadata
from src.utils.file_utils import MODELS_DIR


class TensorApi(UniqueEnum):
    NONE = 'none'
    TENSORFLOW = 'tensorflow'
    KERAS = 'keras'
    TF_LITE = 'tf.lite'
    TENSOR_RT = 'trt'

    def __str__(self):
        return self.name


class AbstractTensorModel(ABC):

    def __init__(self, name, metadata):
        # name
        assert name is not None and isinstance(name, str)
        self.name = name

        # metadata
        assert metadata is not None and isinstance(metadata, Metadata)
        self.metadata = metadata

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

    # Directory

    def file_dir(self, epoch):
        result = os.path.join(MODELS_DIR, self.filename_no_ext(epoch))
        if not os.path.exists(result):
            os.makedirs(result)
        return result

    # Filename

    def filename_no_ext(self, epoch):
        return '%s_%03d' % (self.name, epoch)

    # .h5 - architecture/weights/optimizer

    def filename_h5(self, epoch):
        return '%s%s' % (self.filename_no_ext(epoch), EXTENSION_H5)

    def filepath_h5(self, epoch):
        return os.path.join(self.file_dir(epoch), self.filename_h5(epoch))

    # .h5 - architecture/weights

    def filename_no_opt_h5(self, epoch):
        return '%s%s' % (self.filename_no_ext(epoch), EXTENSION_ARCH_WEIGHTS_H5)

    def filepath_no_opt_h5(self, epoch):
        return os.path.join(self.file_dir(epoch), self.filename_no_opt_h5(epoch))

    # .h5 - weights

    def filename_weights_h5(self, epoch):
        return '%s%s' % (self.filename_no_ext(epoch), EXTENSION_WEIGHTS_H5)

    def filepath_weights_h5(self, epoch):
        return os.path.join(self.file_dir(epoch), self.filename_weights_h5(epoch))

    # .pb

    def filename_pb(self, epoch):
        return '%s%s' % (self.filename_no_ext(epoch), EXTENSION_PB)

    def filepath_pb(self, epoch):
        return os.path.join(self.file_dir(epoch), self.filename_pb(epoch))

    # .tflite - INT8

    def filename_tflite(self, epoch):
        return '%s%s' % (self.filename_no_ext(epoch), EXTENSION_INT8_TFLITE)

    def filepath_tflite(self, epoch):
        return os.path.join(self.file_dir(epoch), self.filename_tflite(epoch))
