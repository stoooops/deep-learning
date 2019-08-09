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

    def __init__(self, name, metadata, mode=TensorApi.NONE):
        # name
        assert name is not None and isinstance(name, str)
        self.name = name

        # metadata
        assert metadata is not None and isinstance(metadata, Metadata)
        self.metadata = metadata

        # mode
        assert mode is not None and isinstance(mode, TensorApi)
        self.mode = mode

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

    def log_prefix(self):
        return '[%s|%s|%s]' % (self.name, self.metadata.epoch, self.mode)

    # Directory

    def file_dir(self, epoch=None):
        epoch = epoch if epoch is not None else self.metadata.epoch
        result = os.path.join(MODELS_DIR, self.filename_no_ext(epoch=epoch))
        if not os.path.exists(result):
            os.makedirs(result)
        return result

    # Filename

    def filename_no_ext(self, epoch=None):
        epoch = epoch if epoch is not None else self.metadata.epoch
        return '%s_%03d' % (self.name, epoch)

    # .h5 - architecture/weights/optimizer

    def filename_h5(self, epoch=None):
        return self.metadata.filename_h5(epoch=epoch)

    def filepath_h5(self, epoch=None, dir_=None):
        return self.metadata.filepath_h5(epoch=epoch, dir_=dir_)

    # .h5 - architecture/weights

    def filename_no_opt_h5(self, epoch=None):
        return self.metadata.filename_no_opt_h5(epoch=epoch)

    def filepath_no_opt_h5(self, epoch=None, dir_=None):
        return self.metadata.filepath_no_opt_h5(epoch=epoch, dir_=dir_)

    # .h5 - weights

    def filename_weights_h5(self, epoch=None):
        return self.metadata.filename_weights_h5(epoch=epoch)

    def filepath_weights_h5(self, epoch=None, dir_=None):
        return self.metadata.filepath_weights_h5(epoch=epoch, dir_=dir_)

    # .pb

    def filename_pb(self, epoch=None):
        return self.metadata.filename_pb(epoch=epoch)

    def filepath_pb(self, epoch=None, dir_=None):
        return self.metadata.filepath_pb(epoch=epoch, dir_=dir_)

    # .tflite - INT8

    def filename_tflite(self, epoch=None):
        return self.metadata.filename_tflite(epoch=epoch)

    def filepath_tflite(self, epoch=None, dir_=None):
        return self.metadata.filepath_tflite(epoch=epoch, dir_=dir_)


class AbstractTensorModelSaver(ABC):

    @abstractmethod
    def save(self, filepath):
        pass

    @abstractmethod
    def save_weights(self, filepath):
        pass
