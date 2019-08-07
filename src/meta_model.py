#!/usr/bin/env python

import os

import tensorflow as tf
from tensorflow import keras

import numpy as np

from src.utils.logger import HuliLogging
from src.utils.file_utils import MODELS_DIR

logger = HuliLogging.get_logger(__name__)


ERROR_TFLITE_FILE_NOT_FOUND = 10001
ERROR_H5_FILE_NOT_FOUND = 10002
ERROR_NOTHING_LOADED = 10003
ERROR_INVALID_STATE = 10004  # bug

EXTENSION_H5 = '.h5'
EXTENSION_TFLITE = '_int8.tflite'

UNKNOWN_EPOCH = -1

MODE_NONE = 'none'
MODE_KERAS_MODEL = 'keras'
MODE_TFLITE_INTERPRETER = 'tflite'


def _scaled_to_real(scaled_value, mean, std):
    return (scaled_value - mean) * std


def _real_to_scaled(real_value, mean, std):
    return (real_value / std) + mean


class MetaModel:

    def __init__(self, name, epoch=UNKNOWN_EPOCH, keras_model=None):
        self.name = name
        self.epoch = epoch
        self.keras_model = keras_model
        if self.keras_model is not None:
            self.mode = MODE_KERAS_MODEL
        else:
            self.mode = MODE_NONE

        self.tflite_interpreter = None
        self.tflite_input_detail = self.tflite_in_mean = self.tflite_in_std = self.tflite_in_index = None
        self.tflite_output_detail = self.tflite_out_mean = self.tflite_out_std = self.tflite_out_index = None
        self.tflite_quantize_in = self.tflite_quantize_out = None
        self.resized = False

    def compile(self, *args, **kwargs):
        assert self.keras_model is not None
        return self.keras_model.compile(*args, **kwargs)

    def fit(self, *argv, **kwargs):
        assert self.keras_model is not None
        result = self.keras_model.fit(*argv, **kwargs)
        self.epoch = kwargs.get('epochs', 1)
        return result

    def evaluate(self, *args, **kwargs):
        assert self.keras_model is not None
        return self.keras_model.evaluate(*args, **kwargs)

    def predict(self, x, **kwargs):
        # ERROR
        if self.mode == MODE_NONE:
            return ERROR_NOTHING_LOADED, None

        # keras
        elif self.mode == MODE_KERAS_MODEL:
            return 0, self.keras_model.predict(x, **kwargs)

        # tflite
        elif self.mode == MODE_TFLITE_INTERPRETER:
            assert len(kwargs) == 0
            quantized_x = self.tflite_quantize_in(x)

            result = []
            for i in range(len(x)):
                self.tflite_interpreter.set_tensor(self.tflite_in_index, quantized_x[i:i + 1])
                self.tflite_interpreter.invoke()
                quantized_y = self.tflite_interpreter.get_tensor(self.tflite_out_index)

                y = self.tflite_quantize_out(quantized_y)
                result.append(y)

            return 0, (np.array(result) if len(result) > 1 else result[0])

        # ERROR
        else:
            return ERROR_INVALID_STATE, None

    def load_keras_model(self):
        if self.keras_model is not None:
            return 0

        self.unload_tflite_interpreter()
        filepath = self.filepath_h5()
        self.keras_model = keras.models.load_model(filepath)
        self.mode = MODE_KERAS_MODEL

        return 0

    def unload_keras_model(self):
        logger.info('%s Deleting keras model...', self.name)
        del self.keras_model
        self.keras_model = None

    def reload_keras_model(self):
        assert self.keras_model is not None
        logger.info('%s Reloading keras model...', self.name)
        del self.keras_model
        self.keras_model = None
        return self.load_keras_model()

    def load_tflite_interpreter(self):
        if self.tflite_interpreter is not None:
            return 0

        self.unload_keras_model()
        tflite_filepath = self.filepath_tflite()
        if os.path.exists(tflite_filepath):
            self.tflite_interpreter = tf.lite.Interpreter(model_path=tflite_filepath)
            self.tflite_interpreter.allocate_tensors()

            self.tflite_input_detail = self.tflite_interpreter.get_input_details()[0]
            self.tflite_in_std, self.tflite_in_mean = self.tflite_input_detail['quantization']
            self.tflite_in_index = self.tflite_input_detail['index']

            self.tflite_output_detail = self.tflite_interpreter.get_output_details()[0]
            self.tflite_out_std, self.tflite_out_mean = self.tflite_output_detail['quantization']
            self.tflite_out_index = self.tflite_output_detail['index']

            def quantize_in(real_value):
                return _real_to_scaled(real_value, self.tflite_in_mean, self.tflite_in_std).astype(np.uint8)

            def quantize_out(real_value):
                return _scaled_to_real(real_value.astype(np.float32), self.tflite_out_mean, self.tflite_out_std)

            self.tflite_quantize_in = quantize_in
            self.tflite_quantize_out = quantize_out

            self.mode = MODE_TFLITE_INTERPRETER
            return 0
        else:
            return ERROR_TFLITE_FILE_NOT_FOUND

    def unload_tflite_interpreter(self):
        if self.tflite_interpreter is not None:
            logger.info('%s Deleting tflite interpreter...', self.name)
            del self.tflite_interpreter
            self.tflite_interpreter = None
            self.tflite_input_detail = self.tflite_in_mean = self.tflite_in_std = self.tflite_in_index = None
            self.tflite_output_detail = self.tflite_out_mean = self.tflite_out_std = self.tflite_out_index = None
            self.tflite_quantize_in = self.tflite_quantize_out = None

    def save(self, convert_tflite=False, representative_data=None, **kwargs):
        # save keras model
        assert self.keras_model is not None
        h5_filepath = self.filepath_h5()
        logger.info('Saving %s...', h5_filepath)
        self.keras_model.save(h5_filepath, **kwargs)

        # convert to tflite and save
        assert convert_tflite == (representative_data is not None)
        if convert_tflite:
            ret = self.convert_tflite(representative_data)
        else:
            ret = 0
        return ret

    def convert_tflite(self, representative_data):
        h5_filepath = self.filepath_h5()
        if not os.path.exists(h5_filepath):
            return ERROR_H5_FILE_NOT_FOUND
        converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(h5_filepath)

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

        logger.info('%s Converting to tflite model...', self.name)
        tflite_model = converter.convert()
        tflite_filepath = self.filepath_tflite()
        logger.info('%s Writing tflite model to %s...', self.name, tflite_filepath)
        with open(tflite_filepath, 'wb') as o_:
            o_.write(tflite_model)

        if self.keras_model is not None:
            # now we need to reload the keras model, else fit() function will fail with strange errors
            self.reload_keras_model()

        return 0

    def summary(self, *args, **kwargs):
        assert self.keras_model is not None
        return self.keras_model.summary(*args, **kwargs)

    def filename_no_ext(self):
        return '%s_%03d' % (self.name, self.epoch)

    def filename_h5(self):
        return '%s%s' % (self.filename_no_ext(), EXTENSION_H5)

    def filename_tflite(self):
        return '%s%s' % (self.filename_no_ext(), EXTENSION_TFLITE)

    def filepath_h5(self):
        return os.path.join(MODELS_DIR, self.filename_h5())

    def filepath_tflite(self):
        return os.path.join(MODELS_DIR, self.filename_tflite())

    @staticmethod
    def from_h5(name, epoch):
        model = MetaModel(name, epoch=epoch)
        model.load_keras_model()
        return model

    @staticmethod
    def from_tflite(name, epoch):
        model = MetaModel(name, epoch=epoch)
        model.load_tflite_interpreter()
        return model
