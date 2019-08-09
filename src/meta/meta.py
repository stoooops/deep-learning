#!/usr/bin/env python

import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.meta.constants import UNKNOWN_EPOCH
from src.meta.errors import *
from src.meta.keras import KerasModel
from src.meta.metadata import Metadata
from src.meta.tensorflow import TensorFlowModel
from src.meta.tflite import TfLiteModel
from src.meta.tensor_apis import AbstractTensorModel, TensorApi
from src.utils.cuda_utils import gpu_info
from src.utils.file_utils import EXTENSION_PB, EXTENSION_INT8_TFLITE
from src.utils.logger import Logging


logger = Logging.get_logger(__name__)

SUPPORTED_MODES = [TensorApi.NONE, TensorApi.KERAS, TensorApi.TENSORFLOW, TensorApi.TF_LITE]


class MetaModel(AbstractTensorModel):

    def __init__(self, name, metadata, delegate=None, f_construct_keras_model=None):
        super().__init__(name, metadata, mode=TensorApi.NONE)

        assert delegate is None or isinstance(delegate, AbstractTensorModel)
        self.delegate = None
        if delegate is not None:
            self.attach_delegate(delegate)

        assert f_construct_keras_model is None or callable(f_construct_keras_model)
        self.f_construct_keras_model = f_construct_keras_model

    def attach_delegate(self, delegate):
        """
        :type delegate: AbstractTensorModel
        """
        assert self.mode == TensorApi.NONE
        assert isinstance(delegate, AbstractTensorModel), 'Expected AbstractTensorModel but got: %s' % delegate

        if delegate.mode == TensorApi.KERAS:
            assert isinstance(delegate, KerasModel)
        elif delegate.mode == TensorApi.TENSORFLOW:
            assert isinstance(delegate, TensorFlowModel)
        elif delegate.mode == TensorApi.TF_LITE:
            assert isinstance(delegate, TfLiteModel)
        else:
            assert 0 == 1, 'Unexpected delegate mode: %s' % delegate.mode

        self.delegate = delegate
        self.mode = delegate.mode

        return 0

    def compile(self, *argv, **kwargs):
        return self.delegate.compile(*argv, **kwargs)

    def fit(self, *argv, **kwargs):
        # get epoch
        epochs = kwargs.get('epochs', 1)
        # get initial_epoch, falling back to self.metadata.epoch if not set
        initial_epoch = kwargs.get('initial_epoch')
        if initial_epoch is None:
            initial_epoch = self.metadata.epoch
            logger.debug('%s Assuming initial_epoch = %d', self.log_prefix(), initial_epoch)
            kwargs['initial_epoch'] = initial_epoch

        # validate input
        if initial_epoch >= epochs:
            logger.error('Bad input for state: initial_epoch = %d >= %d = epochs', initial_epoch, epochs)
            return ERROR_TF_META_BAD_INPUT, None
        if self.metadata.epoch != UNKNOWN_EPOCH and self.metadata.epoch != initial_epoch:
            logger.error('Bad input for state: self.metadata.epoch = %d != %d = initial_epoch', self.metadata.epoch,
                         initial_epoch)
            return ERROR_TF_META_BAD_INPUT, None

        ret, history = self.delegate.fit(*argv, **kwargs)
        if ret != 0:
            return ret, history

        self.metadata.update_epoch(epochs)
        return ret, history

    def evaluate(self, *argv, **kwargs):
        return self.delegate.evaluate(*argv, **kwargs)

    def predict(self, *argv, **kwargs):
        return self.delegate.predict(*argv, **kwargs)

    def save(self, **kwargs):
        assert self.mode == TensorApi.KERAS  # TODO improve state logic

        filepath = self.filepath_h5() if self.mode == TensorApi.KERAS else None
        logger.debug('%s Saving keras model to %s...', self.log_prefix(), filepath)
        ret = self.delegate.save(filepath, **kwargs)
        if ret != 0:
            return ret

        if self.mode == TensorApi.KERAS:
            # save weights
            filepath_weights = self.filepath_weights_h5()
            logger.debug('%s Saving keras model weights to %s...', self.log_prefix(), filepath_weights)
            ret = self.delegate.save_weights(filepath_weights, **kwargs)
            if ret != 0:
                return ret

            # unless explicitly passed in as include_optimizer=True, then also store without the optimizer info
            if not kwargs.get('include_optimizer', False):
                filepath_no_opt = self.filepath_no_opt_h5()
                logger.debug('%s Saving keras model without optimizer to %s...', self.log_prefix(), filepath_no_opt)
                kwargs['include_optimizer'] = True
                ret = self.delegate.save(filepath_no_opt, **kwargs)
                if ret != 0:
                    return ret

        return ret

    def dump(self):
        gpu_info(print_prefix=self.log_prefix())

        logger.debug('%s mode = %s', self.log_prefix(), self.mode)
        self.metadata.dump(prefix=self.log_prefix())

        ret = 0
        if self.mode != TensorApi.NONE:
            ret = self.delegate.dump()
        return ret

    # New APIs

    def log_prefix(self):
        return '[%s|%s|%s]' % (self.name, self.metadata.epoch, self.mode)

    def summary(self, *args, **kwargs):
        if self.mode != TensorApi.KERAS:
            logger.error('%s summary() not available in mode %s', self.log_prefix(), self.mode)
            return ERROR_TF_META_WRONG_MODE
        return self.delegate.summary(*args, **kwargs)

    def reload(self):
        assert self.mode == TensorApi.KERAS
        if self.mode == TensorApi.KERAS:
            ret = self.delegate.restart_session()
            if ret != 0:
                return ret
        else:
            return ERROR_TF_META_UNIMPLEMENTED

        return 0

    def init_keras(self, f_construct_keras_model=None):
        # TODO this big block needs to move to a function of KerasModel
        #  so it can restart the underlying keras model on its own
        assert self.delegate is None

        logger.debug('%s Clearing keras session...', self.log_prefix())
        keras.backend.clear_session()

        if f_construct_keras_model is None:
            # No constructor passed, loading from h5
            filepath_h5 = self.filepath_h5()
            logger.debug('%s Initializing keras from %s...', self.log_prefix(), filepath_h5)
            ret, keras_model = KerasModel.load(self.name, self.metadata, filepath_h5)
            if ret != 0:
                return ret
        else:
            # constructor passed, running create func and loading weights
            logger.debug('%s Initializing keras model from factory function...', self.log_prefix())
            keras_model = f_construct_keras_model()

            if self.metadata.epoch > 0:
                # weights
                filepath_weights_h5 = self.filepath_weights_h5()
                logger.debug('%s Loading keras model weights from %s...', self.log_prefix(), filepath_weights_h5)
                keras_model.load_weights(filepath_weights_h5)
            else:
                logger.debug('%s Skipping loading weights since epoch is 0', self.log_prefix())

            keras_model = KerasModel(self.name, self.metadata, keras_model,
                                     f_construct_keras_model=f_construct_keras_model)

        logger.debug('%s Attaching keras delegate...', self.log_prefix())
        self.attach_delegate(keras_model)
        return 0

    def init_tflite(self):
        ret, tflite_model = TfLiteModel.load(self.name, self.metadata, self.filepath_tflite())
        if ret != 0:
            return ret

        self.attach_delegate(tflite_model)
        return 0

    def init_pb(self):
        assert self.metadata.input_names is not None and self.metadata.output_names is not None
        ret, tf_model = TensorFlowModel.load(self.name, self.metadata, self.filepath_pb())
        if ret != 0:
            return ret

        self.attach_delegate(tf_model)
        return 0

    def change_mode(self, mode):
        if self.mode == mode:
            return 0

        if mode == TensorApi.NONE:
            del self.delegate
            self.delegate = None
            self.mode = mode
            return 0

        ret = self.change_mode(TensorApi.NONE)
        if ret != 0:
            return ret

        if mode == TensorApi.KERAS:
            ret = self.init_keras(f_construct_keras_model=self.f_construct_keras_model)
        elif mode == TensorApi.TENSORFLOW:
            ret = self.init_pb()
        elif mode == TensorApi.TF_LITE:
            ret = self.init_tflite()
        else:
            ret = ERROR_TF_META_BAD_INPUT

        return ret

    def freeze_graph(self, filepath_pb=None):
        if self.mode == TensorApi.KERAS:
            filepath_pb = filepath_pb or self.filepath_pb()
            return self.delegate.freeze_graph(filepath_pb)
        else:
            return ERROR_TF_META_UNIMPLEMENTED

    def save_to(self, mode, representative_data=None):
        if (mode == TensorApi.TF_LITE) != (representative_data is not None):
            logger.error('mode is %s but representative data is %sNone', mode,
                         'not ' if representative_data is not None else '')
            return ERROR_TF_META_BAD_INPUT

        logger.debug('%s Saving mode %s via mode %s...', self.log_prefix(), mode, self.mode)

        if self.mode == TensorApi.KERAS:
            if mode == TensorApi.TENSORFLOW:
                logger.debug('%s Converting to %s...', self.log_prefix(), EXTENSION_PB)
                return self.freeze_graph(self.filepath_pb())

            elif mode == TensorApi.TF_LITE:
                logger.debug('%s Converting to %s...', self.log_prefix(), EXTENSION_INT8_TFLITE)
                self.delegate.save(self.filepath_tflite())

            else:
                assert 1 == 0, 'Unexpected mode: %s' % mode

    def save_all(self, representative_data=None):
        # Save keras file first
        save_order = [TensorApi.KERAS] + [t for t in SUPPORTED_MODES if t is not TensorApi.KERAS]
        for mode in save_order:
            if mode == TensorApi.NONE:
                continue

            if mode == self.mode:
                logger.debug('%s Saving...', self.log_prefix())
                ret = self.save()

            elif mode == TensorApi.TENSORFLOW:
                logger.debug('%s Saving to %s...', self.log_prefix(), mode)
                ret = self.save_to(mode)

            elif mode == TensorApi.TF_LITE:
                logger.debug('%s Saving to %s...', self.log_prefix(), mode)
                ret = self.save_to(mode, representative_data=representative_data)

            else:
                ret = ERROR_TF_META_WRONG_MODE

            if ret != 0:
                return ret
        return 0


class MetaModelFactory:

    @staticmethod
    def from_h5(name, epoch):
        metadata = Metadata(name, epoch=epoch)
        model = MetaModel(name, metadata)
        ret = model.init_keras()
        if ret != 0:
            return ret, None
        return 0, model

    @staticmethod
    def from_weights_h5(name, epoch, f_construct_keras_model):
        metadata = Metadata(name, epoch=epoch)
        model = MetaModel(name, metadata, f_construct_keras_model=f_construct_keras_model)
        ret = model.init_keras(f_construct_keras_model=f_construct_keras_model)
        if ret != 0:
            return ret, None
        return 0, model

    @staticmethod
    def from_pb(name, metadata):
        model = MetaModel(name, metadata)
        ret = model.init_pb()
        if ret != 0:
            return ret, None
        return 0, model

    @staticmethod
    def from_tflite(name, epoch):
        metadata = Metadata(name, epoch=epoch)
        model = MetaModel(name, metadata)
        ret = model.init_tflite()
        if ret != 0:
            return ret, None
        return 0, model
