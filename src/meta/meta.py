#!/usr/bin/env python

import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.meta.constants import UNKNOWN_EPOCH
from src.meta.errors import *
from src.meta.keras import KerasModel
from src.meta.tflite import TfLiteModel
from src.meta.tensor_apis import AbstractTensorModel, TensorApi
from src.utils.logger import HuliLogging


logger = HuliLogging.get_logger(__name__)

SUPPORTED_MODES = [TensorApi.NONE, TensorApi.KERAS, TensorApi.TF_LITE]


class MetaModel(AbstractTensorModel):

    def __init__(self, name, epoch=UNKNOWN_EPOCH):
        super().__init__(name)
        self.mode = TensorApi.NONE

        assert isinstance(epoch, int) and (epoch == UNKNOWN_EPOCH or epoch >= 1)
        self.epoch = epoch

        self.delegate = None

    def attach_delegate(self, delegate):
        """
        :type delegate: AbstractTensorModel
        """
        assert self.mode == TensorApi.NONE
        assert isinstance(delegate, AbstractTensorModel)

        if delegate.mode == TensorApi.KERAS:
            assert isinstance(delegate, KerasModel)
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
        # get initial_epoch, falling back to self.epoch if not sef
        initial_epoch = kwargs.get('initial_epoch')
        if initial_epoch is None:
            initial_epoch = self.epoch
            logger.debug('%s Assuming initial_epoch = %d', initial_epoch)
            kwargs['initial_epoch'] = initial_epoch

        # validate input
        if initial_epoch >= epochs:
            logger.error('Bad input for state: initial_epoch = %d >= %d = epochs', initial_epoch, epochs)
            return ERROR_TF_META_BAD_INPUT, None
        if self.epoch != UNKNOWN_EPOCH and self.epoch != initial_epoch:
            logger.error('Bad input for state: self.epoch = %d != %d = initial_epoch', self.epoch, initial_epoch)
            return ERROR_TF_META_BAD_INPUT, None

        ret, history = self.delegate.fit(*argv, **kwargs)
        if ret != 0:
            return ret, history

        self.epoch = epochs
        return ret, history

    def evaluate(self, *argv, **kwargs):
        return self.delegate.evaluate(*argv, **kwargs)

    def predict(self, *argv, **kwargs):
        return self.delegate.predict(*argv, **kwargs)

    def save(self, **kwargs):
        filepath = self.filename_h5(self.epoch) if self.mode == TensorApi.KERAS else None
        return self.delegate.save(filepath, **kwargs)

    def dump(self):
        logger.debug('%s mode = %s', self.name, self.mode)
        logger.debug('%s epoch = %s', self.name, self.epoch)
        if self.mode != TensorApi.NONE:
            self.delegate.dump()

    # New APIs

    def reload(self):
        assert self.mode == TensorApi.KERAS
        before = self.mode

        logger.debug('%s Deleting existing delegate...', self.name)
        del self.delegate
        self.delegate = None
        self.mode = TensorApi.NONE

        if before == TensorApi.KERAS:
            ret, keras_model = TensorModelLoader.load_keras_model(self.name, self.epoch, self.filepath_h5(self.epoch))
            if ret != 0:
                return ERROR_TF_META_FAILED_RELOAD
            self.delegate = keras_model
        else:
            return ERROR_TF_META_UNIMPLEMENTED

        self.mode = before

        return 0

    def freeze_session(self):
        if self.mode == TensorApi.KERAS:
            return self.delegate.freeze_session()
        else:
            return ERROR_TF_META_UNIMPLEMENTED

    def save_to(self, mode, representative_data=None):
        if (mode == TensorApi.TF_LITE) != (representative_data is not None):
            logger.error('mode is %s but representative data is %sNone', mode,
                         'not ' if representative_data is not None else '')
            return ERROR_TF_META_BAD_INPUT

        if self.mode == TensorApi.KERAS:
            if mode == TensorApi.TF_LITE:
                return MetaModelModeConverter(self).save_tflite(representative_data)

    def save_all(self, representative_data=None):
        for mode in [m for m in SUPPORTED_MODES if m != TensorApi.NONE]:
            if mode == self.mode:
                ret = self.save()
            elif mode == TensorApi.TENSOR_FLOW:
                ret = self.freeze_session()
            elif mode == TensorApi.TF_LITE:
                ret = self.save_to(mode, representative_data=representative_data)
            else:
                ret = ERROR_TF_META_WRONG_MODE
            if ret != 0:
                return ret
        return 0


def _scaled_to_real(scaled_value, mean, std):
    return (scaled_value - mean) * std


def _real_to_scaled(real_value, mean, std):
    return (real_value / std) + mean

class TensorModelLoader:

    @staticmethod
    def load_keras_model(name, epoch, filepath): #, reload=False):
        # if not reload and self.mode == TensorApi.KERAS:
        #     return 0

        #filepath = self.filepath_h5(epoch)
        logger.debug('%s Loading keras model from %s...', name, filepath)
        try:
            keras_model = keras.models.load_model(filepath)
        except IOError as e:
            logger.exception(e)
            return ERROR_TF_META_CAUGHT_EXCEPTION
        except Exception as e:
            logger.exception(e)
            return ERROR_TF_META_CAUGHT_EXCEPTION

        result = KerasModel(name, keras_model, epoch=epoch)
        # self.mode = TensorApi.KERAS
        # self.epoch = epoch

        return 0, result

    @staticmethod
    def load_tflite_interpreter(name, epoch, filepath):
        # if not reload and self.mode == TensorApi.TF_LITE:
        #     return 0

        #filepath = self.filepath_tflite()
        logger.debug('%s Loading tflite interpreter from %s...', name, filepath)
        if os.path.exists(filepath):
            tflite_interpreter = tf.lite.Interpreter(model_path=filepath)
            tflite_interpreter.allocate_tensors()

            tflite_input_detail = tflite_interpreter.get_input_details()[0]
            tflite_in_std, tflite_in_mean = tflite_input_detail['quantization']
            tflite_in_index = tflite_input_detail['index']

            tflite_output_detail = tflite_interpreter.get_output_details()[0]
            tflite_out_std, tflite_out_mean = tflite_output_detail['quantization']
            tflite_out_index = tflite_output_detail['index']

            def quantize_in(real_value):
                return _real_to_scaled(real_value, tflite_in_mean, tflite_in_std).astype(np.uint8)

            def quantize_out(real_value):
                return _scaled_to_real(real_value.astype(np.float32), tflite_out_mean, tflite_out_std)

            result = TfLiteModel(name, tflite_interpreter, quantize_in, quantize_out, epoch=epoch)

            #self.mode = MODE_TFLITE_INTERPRETER
            return 0, result
        else:
            return ERROR_TF_META_FILE_NOT_FOUND,None


class MetaModelFactory:

    @staticmethod
    def from_h5(name, epoch):
        model = MetaModel(name, epoch=epoch)
        ret, keras_model = TensorModelLoader.load_keras_model(name, epoch, model.filepath_h5(model.epoch))
        if ret != 0:
            return ret, None

        model.attach_delegate(keras_model)
        return 0, model

    @staticmethod
    def from_tflite(name, epoch):
        model = MetaModel(name, epoch=epoch)
        ret, tflite_model = TensorModelLoader.load_tflite_interpreter(name, epoch, model.filepath_tflite(model.epoch))
        if ret != 0:
            return ret, None

        model.attach_delegate(tflite_model)
        return 0, model


class MetaModelModeConverter:

    def __init__(self, meta_model):
        assert isinstance(meta_model, MetaModel)
        self.meta_model = meta_model

    def save_tflite(self, representative_data):
        h5_filepath = self.meta_model.filepath_h5(self.meta_model.epoch)
        if not os.path.exists(h5_filepath):
            return ERROR_TF_META_FILE_NOT_FOUND

        converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(h5_filepath)
        # above call clears the keras session, so we need to reload our model
        if self.meta_model.mode == TensorApi.KERAS:
            self.meta_model.reload()

        def representative_dataset_gen():
            for i in range(1000):
                yield [representative_data[i: i + 1].astype(np.float32)]

        converter.representative_dataset = representative_dataset_gen

        # throws error if conversion not available
        target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        if tf.__version__[0] == '2':
            converter.target_spec.supported_ops = target_ops
        else:
            converter.target_ops = target_ops
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        converter.optimizations = [tf.lite.Optimize.DEFAULT]  # seems that result file has same size no matter what

        logger.info('%s Converting to tflite INT8 model...', self.meta_model.name)
        tflite_model = converter.convert()
        tflite_filepath = self.meta_model.filepath_tflite(self.meta_model.epoch)
        logger.info('%s Saving tflite model to %s...', self.meta_model.name, tflite_filepath)
        with open(tflite_filepath, 'wb') as o_:
            o_.write(tflite_model)

        return 0
