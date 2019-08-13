#!/usr/bin/env python

import os
import numpy as np
import tensorflow as tf

from .errors import ERROR_TF_LITE_FILE_NOT_FOUND
from src.meta.inference import InferenceModel
from src.meta.metadata import Metadata
from src.utils.logger import Logging
from src.utils import file_utils

logger = Logging.get_logger(__name__)


class TfLiteModel(InferenceModel):

    def __init__(self, metadata, tflite_interpreter, quantize_in, quantize_out):
        """
        :type tflite_interpreter: tensorflow.lite.Interpreter
        :type metadata: Metadata
        """
        assert metadata is not None and isinstance(metadata, Metadata)
        self.name = metadata.name
        self.metadata = metadata

        # tf.lite Interpreter
        assert tflite_interpreter is not None and isinstance(tflite_interpreter, tf.lite.Interpreter)
        self.tflite_interpreter = tflite_interpreter
        self.in_index = self.tflite_interpreter.get_input_details()[0]['index']
        self.out_index = self.tflite_interpreter.get_output_details()[0]['index']

        self.f_quantize_in = quantize_in
        self.f_quantize_out = quantize_out

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

    def dump(self):
        input_detail = self.tflite_interpreter.get_input_details()[0]
        in_std, in_mean = input_detail['quantization']
        in_index = input_detail['index']
        logger.debug('Input mean, std, index: %s, %s, %s', in_mean, in_std, in_index)

        output_detail = self.tflite_interpreter.get_output_details()[0]
        out_std, out_mean = output_detail['quantization']
        out_index = output_detail['index']
        logger.debug('Output mean, std, index: %s, %s, %s', out_mean, out_std, out_index)

        return 0

    @staticmethod
    def from_tflite(filepath_md, filepath_tflite):
        assert os.path.exists(filepath_md)
        assert os.path.splitext(filepath_md)[1] == file_utils.EXTENSION_MD
        assert os.path.exists(filepath_tflite)
        assert os.path.splitext(filepath_tflite)[1] == file_utils.EXTENSION_TFLITE

        is_edgetpu = ('edgetpu' in filepath_tflite)

        logger.debug('Loading metadata from %s...', filepath_md)
        ret, metadata = Metadata.from_md(filepath_md)
        if ret != 0:
            return ret

        logger.debug('%s Loading tflite interpreter from %s...', metadata.name, filepath_tflite)
        if os.path.exists(filepath_tflite):
            from tensorflow.lite.python.interpreter import Interpreter
            if is_edgetpu:
                from tensorflow.lite.python.interpreter import load_delegate
                experimental_lib = 'libedgetpu.so.1.0'
                logger.info('Using experimental library %s...', 'libedgetpu.so.1.0')
                tflite_interpreter = Interpreter(model_path=filepath_tflite,
                                                 experimental_delegates=[load_delegate(experimental_lib)])
            else:
                tflite_interpreter = Interpreter(model_path=filepath_tflite)
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

            result = TfLiteModel(metadata, tflite_interpreter, quantize_in, quantize_out)

            return 0, result
        else:
            return ERROR_TF_LITE_FILE_NOT_FOUND, None
