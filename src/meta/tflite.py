#!/usr/bin/env python

import os
import numpy as np
import tensorflow as tf

from src.meta.constants import UNKNOWN_EPOCH
from src.meta.errors import *
from src.meta.tensor_apis import AbstractTensorModel, TensorApi
from src.utils.logger import HuliLogging

logger = HuliLogging.get_logger(__name__)


class TfLiteModel(AbstractTensorModel):

    def __init__(self, name, tflite_interpreter, quantize_in, quantize_out, epoch=UNKNOWN_EPOCH):
        """
        :type name: str
        :type tflite_interpreter: tensorflow.lite.Interpreter
        :type epoch: int
        """
        super().__init__(name)

        assert isinstance(epoch, int) and (epoch == UNKNOWN_EPOCH or epoch >= 1)
        self.epoch = epoch

        # tf.lite Interpreter
        assert tflite_interpreter is not None and isinstance(tflite_interpreter, tf.lite.Interpreter)
        self.tflite_interpreter = tflite_interpreter
        self.in_index = self.tflite_interpreter.get_input_details()[0]['index']
        self.out_index = self.tflite_interpreter.get_output_details()[0]['index']

        self.f_quantize_in = quantize_in
        self.f_quantize_out = quantize_out

        self.mode = TensorApi.TF_LITE

    def compile(self):
        return ERROR_TF_META_UNIMPLEMENTED

    def fit(self):
        return ERROR_TF_META_UNIMPLEMENTED

    def evaluate(self):
        return ERROR_TF_META_UNIMPLEMENTED

    def predict(self, *argv, **kwargs):
        assert len(argv) == 1
        x = argv[0]
        assert len(kwargs) == 0
        quantized_x = self.f_quantize_in(x)

        result = []
        for i in range(len(x)):
            self.tflite_interpreter.set_tensor(self.in_index, quantized_x[i:i + 1])
            self.tflite_interpreter.invoke()
            quantized_y = self.tflite_interpreter.get_tensor(self.out_index)

            y = self.f_quantize_out(quantized_y)
            result.append(y)

        return 0, (np.array(result) if len(result) > 1 else result[0])

    def save(self):
        return ERROR_TF_META_UNIMPLEMENTED

    def dump(self):
        input_detail = self.tflite_interpreter.get_input_details()[0]
        in_std, in_mean = input_detail['quantization']
        in_index = input_detail['index']
        logger.debug('%s Input mean, std, index: %s, %s, %s', self.name, in_mean, in_std, in_index)

        output_detail = self.tflite_interpreter.get_output_details()[0]
        out_std, out_mean = output_detail['quantization']
        out_index = output_detail['index']
        logger.debug('%s Output mean, std, index: %s, %s, %s', self.name, out_mean, out_std, out_index)

    @staticmethod
    def load(name, epoch, filepath):
        logger.debug('%s Loading tflite interpreter from %s...', name, filepath)
        if os.path.exists(filepath):
            tflite_interpreter = tf.lite.Interpreter(model_path=filepath)
            tflite_interpreter.allocate_tensors()

            tflite_input_detail = tflite_interpreter.get_input_details()[0]
            tflite_in_std, tflite_in_mean = tflite_input_detail['quantization']

            tflite_output_detail = tflite_interpreter.get_output_details()[0]
            tflite_out_std, tflite_out_mean = tflite_output_detail['quantization']

            def _scaled_to_real(scaled_value, mean, std):
                return (scaled_value - mean) * std

            def _real_to_scaled(real_value, mean, std):
                return (real_value / std) + mean

            def quantize_in(real_value):
                return _real_to_scaled(real_value, tflite_in_mean, tflite_in_std).astype(np.uint8)

            def quantize_out(real_value):
                return _scaled_to_real(real_value.astype(np.float32), tflite_out_mean, tflite_out_std)

            result = TfLiteModel(name, tflite_interpreter, quantize_in, quantize_out, epoch=epoch)

            return 0, result
        else:
            return ERROR_TF_META_FILE_NOT_FOUND, None
