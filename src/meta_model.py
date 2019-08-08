#!/usr/bin/env python

import os
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.compiler.tensorrt import trt_convert as trt

import numpy as np

from src.utils.logger import HuliLogging
from src.utils.file_utils import MODELS_DIR, TMP_DIR

logger = HuliLogging.get_logger(__name__)


ERROR_TFLITE_FILE_NOT_FOUND = 10001
ERROR_H5_FILE_NOT_FOUND = 10002
ERROR_NOTHING_LOADED = 10003
ERROR_INVALID_STATE = 10004  # bug

EXTENSION_H5 = '.h5'
EXTENSION_PB = '.pb'
EXTENSION_TFLITE = '_int8.tflite'

UNKNOWN_EPOCH = -1

MODE_NONE = 'none'
MODE_KERAS_MODEL = 'keras'
MODE_TFLITE_INTERPRETER = 'tflite'
MODE_TRT = 'trt'


def _scaled_to_real(scaled_value, mean, std):
    return (scaled_value - mean) * std


def _real_to_scaled(real_value, mean, std):
    return (real_value / std) + mean


class MetaModel:
    """
    API wrapper around various tensorflow libraries / add-ons. Supports: keras, tflite
    """

    def __init__(self, name, epoch=UNKNOWN_EPOCH, keras_model=None):
        self.name = name
        self.epoch = epoch

        # Keras
        self.keras_model = keras_model
        if self.keras_model is not None:
            self.mode = MODE_KERAS_MODEL
            self.keras_input = self.keras_model.input
            self.keras_input_name = self.keras_input.name.split(':')[0]
            self.keras_output = self.keras_model.output
            self.keras_output_name = self.keras_output.name.split(':')[0]
        else:
            self.mode = MODE_NONE
            self.keras_input = self.keras_input_name = self.keras_output = self.keras_output_name = None
        self.keras_tensorboard_callback = None

        # tflite
        self.tflite_interpreter = None
        self.tflite_input_detail = self.tflite_in_mean = self.tflite_in_std = self.tflite_in_index = None
        self.tflite_output_detail = self.tflite_out_mean = self.tflite_out_std = self.tflite_out_index = None
        self.tflite_quantize_in = self.tflite_quantize_out = None
        self.resized = False

        # TensorRT
        self.trt_model = None

    # TRAIN

    def compile(self, *args, **kwargs):
        assert self.keras_model is not None
        return self.keras_model.compile(*args, **kwargs)

    def fit(self, *argv, **kwargs):
        assert self.keras_model is not None
        if self.keras_tensorboard_callback is None:
            log_dir = os.path.join(TMP_DIR, 'tensorboard/' + datetime.now().strftime("%Y%m%d-%H%M%S") + '_' + self.name)
            self.keras_tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
        callbacks_param = kwargs.get('callbacks', [])
        callbacks_param.append(self.keras_tensorboard_callback)
        kwargs['callbacks'] = callbacks_param
        result = self.keras_model.fit(*argv, **kwargs)
        self.epoch = kwargs.get('epochs', 1)
        return result

    # INFER / TEST

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

    # LOAD KERAS

    def load_keras_model(self):
        if self.keras_model is not None:
            return 0

        self.unload_tflite_interpreter()
        self.unload_trt_model()

        filepath = self.filepath_h5()
        logger.debug('%s Loading keras model from %s...', self.name, filepath)
        self.keras_model = keras.models.load_model(filepath)
        self.keras_input = self.keras_model.input
        self.keras_input_name = self.keras_input.name.split(':')[0]
        self.keras_output = self.keras_model.output
        self.keras_output_name = self.keras_output.name.split(':')[0]
        self.mode = MODE_KERAS_MODEL

        return 0

    def unload_keras_model(self):
        if self.mode == MODE_KERAS_MODEL:
            self.mode = MODE_NONE
        if self.keras_model is not None:
            logger.debug('%s Deleting keras model...', self.name)
            del self.keras_model
            self.keras_model = None

    def reload_keras_model(self):
        assert self.keras_model is not None
        logger.debug('%s Reloading keras model...', self.name)
        self.unload_keras_model()
        return self.load_keras_model()

    def freeze_keras_session(self, keep_var_names=None, output_names=None, clear_devices=True):
        """
        Freezes the state of a session into a pruned computation graph.

        Creates a new computation graph where variable nodes are replaced by
        constants taking their current value in the session. The new graph will be
        pruned so subgraphs that are not necessary to compute the requested
        outputs are removed.
        @param keep_var_names A list of variable names that should not be frozen,
                              or None to freeze all the variables in the graph.
        @param output_names Names of the relevant graph outputs.
        @param clear_devices Remove the device directives from the graph for better portability.
        @return The frozen graph definition.
        """
        # Refer https://stackoverflow.com/a/45466355/2079993
        session = keras.backend.get_session()
        graph = session.graph
        with graph.as_default():
            freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
            # tweaked here
            output_names = output_names or [out.op.name for out in self.keras_model.outputs]
            output_names += [v.op.name for v in tf.global_variables()]
            # Graph -> GraphDef ProtoBuf
            input_graph_def = graph.as_graph_def()
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ""
            frozen_graph = tf.graph_util.convert_variables_to_constants(session, input_graph_def, output_names,
                                                                        freeze_var_names)

            logger.info('%s Saving frozen graph to %s...', self.name, self.filepath_pb())
            tf.train.write_graph(frozen_graph, MODELS_DIR, self.filename_pb(), as_text=False)

            #import ipdb; ipdb.set_trace()
            logger.info('')
        return 0

    # LOAD TFLITE

    def load_tflite_interpreter(self):
        if self.tflite_interpreter is not None:
            return 0

        self.unload_keras_model()
        self.unload_trt_model()

        filepath = self.filepath_tflite()
        logger.debug('%s Loading tflite interpreter from %s...', self.name, filepath)
        if os.path.exists(filepath):
            self.tflite_interpreter = tf.lite.Interpreter(model_path=filepath)
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
        if self.mode == MODE_TFLITE_INTERPRETER:
            self.mode = MODE_NONE
        if self.tflite_interpreter is not None:
            logger.debug('%s Deleting tflite interpreter...', self.name)
            del self.tflite_interpreter
            self.tflite_interpreter = None
            self.tflite_input_detail = self.tflite_in_mean = self.tflite_in_std = self.tflite_in_index = None
            self.tflite_output_detail = self.tflite_out_mean = self.tflite_out_std = self.tflite_out_index = None
            self.tflite_quantize_in = self.tflite_quantize_out = None

    # LOAD TENSORRT

    def load_trt_model(self):
        gpu_mb = 1024*10
        trt_memory_mb = 1024*4
        if self.trt_model is not None:
            return 0

        outputs = [out.op.name for out in self.keras_model.outputs]
        logger.info('OUTPUTS: %s', outputs)
        self.unload_keras_model()
        self.unload_tflite_interpreter()

        with tf.Session() as sess:
            # First deserialize your frozen graph:
            filepath = self.filepath_pb()
            logger.info('%s Loading frozen graph from %s...', self.name, filepath)
            with tf.gfile.GFile(filepath, 'rb') as f:
                frozen_graph = tf.GraphDef()
                frozen_graph.ParseFromString(f.read())

            # Then, we import the graph_def into a new Graph and return it
            with tf.Graph().as_default() as graph:
                # The name var will prefix every op/nodes in your graph
                # Since we load everything in a new graph, this is not needed
                tf.import_graph_def(frozen_graph, name=self.name)

            g = graph
            fg = frozen_graph
            import ipdb; ipdb.set_trace()
            #
            tf_memory_remaining = (gpu_mb - trt_memory_mb) / gpu_mb
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=tf_memory_remaining)
            trt_memory_b = trt_memory_mb * 1024 * 1024

            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)).as_default():
                trt_graph = trt.create_inference_graph(input_graph_def=frozen_graph,
                                                       outputs=outputs,
                                                       max_batch_size=1,
                                                       max_workspace_size_bytes=trt_memory_b,
                                                       precision_mode="INT8")
                logger.info('type: %s', type(trt_graph))

            # Now you can create a TensorRT inference graph from your frozen graph:
            logger.info('%s Converting frozen graph to TRT...', self.name)
            converter = trt.TrtGraphConverter(input_graph_def=frozen_graph,
                                              nodes_blacklist=[self.keras_output_name])  # output nodes
            trt_graph = converter.convert()

            # Import the TensorRT graph into a new graph and run:
            logger.info('%s Importing TRT graph into a new graph and running...', self.name)
            output_node = tf.import_graph_def(trt_graph, return_elements=[self.keras_output_name])
            sess.run(output_node)

    def unload_trt_model(self):
        if self.mode == MODE_TRT:
            self.mode = MODE_NONE
        if self.trt_model is not None:
            logger.debug('%s Deleting TensorRT model...', self.name)
            del self.trt_model
            self.trt_model = None

    # SAVE

    def save(self, convert_tflite=False, representative_data=None, **kwargs):
        # save keras model
        assert self.keras_model is not None
        h5_filepath = self.filepath_h5()
        logger.info('%s Saving keras model to %s...', self.name, h5_filepath)
        self.keras_model.save(h5_filepath, **kwargs)

        # Freeze keras session
        self.freeze_keras_session()

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
        # above call clears the keras session, so we need to reload our model
        if self.keras_model is not None:
            self.reload_keras_model()

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
        logger.info('%s Saving tflite model to %s...', self.name, tflite_filepath)
        with open(tflite_filepath, 'wb') as o_:
            o_.write(tflite_model)

        return 0

    # SUMMARIZE

    def summary(self, *args, **kwargs):
        assert self.keras_model is not None
        return self.keras_model.summary(*args, **kwargs)

    # FILENAMES

    def filename_no_ext(self):
        return '%s_%03d' % (self.name, self.epoch)

    def filename_h5(self):
        return '%s%s' % (self.filename_no_ext(), EXTENSION_H5)

    def filename_pb(self):
        return '%s%s' % (self.filename_no_ext(), EXTENSION_PB)

    def filename_tflite(self):
        return '%s%s' % (self.filename_no_ext(), EXTENSION_TFLITE)

    def filepath_h5(self):
        return os.path.join(MODELS_DIR, self.filename_h5())

    def filepath_pb(self):
        return os.path.join(MODELS_DIR, self.filename_pb())

    def filepath_tflite(self):
        return os.path.join(MODELS_DIR, self.filename_tflite())

    # FACTORIES

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
