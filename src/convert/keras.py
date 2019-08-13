#!/usr/bin/env python


import os

import numpy as np
from src.utils import file_utils
from src.utils.logger import Logging
from src.keras import safe_keras

import tensorflow as tf
from tensorflow import keras


logger = Logging.get_logger(__name__)


def _keras_from_func(f_construct_keras_model, weights_h5):
    keras_model = f_construct_keras_model()

    logger.debug('%s Loading keras model weights from %s...', weights_h5)
    keras_model.load_weights(weights_h5)

    return keras_model


def freeze_graph(input_filepath, output_filepath, f_construct_keras_model=None, keep_var_names=None, output_names=None,
                 clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param filepath Override directory to freeze graph to
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    assert os.path.splitext(input_filepath)[1] == file_utils.EXTENSION_H5, \
        'Unexpected extension for file: %s' % input_filepath
    assert os.path.splitext(output_filepath)[1] == file_utils.EXTENSION_PB, \
        'Unexpected extension for file: %s' % output_filepath

    # restart keras session in learning phase
    logger.warn('Clearing keras session...')
    keras.backend.clear_session()
    logger.warn('Setting learning phase to False')
    keras.backend.set_learning_phase(False)

    # construct keras model
    if f_construct_keras_model is not None:
        # assumes filepath is a weights file
        keras_model = _keras_from_func(f_construct_keras_model, input_filepath)
    else:
        ret, keras_model = safe_keras.load_model(input_filepath)
        if ret != 0:
            return ret

    # Now do freeze

    # Refer https://stackoverflow.com/a/45466355/2079993
    session = tf.compat.v1.keras.backend.get_session()
    graph = session.graph
    with graph.as_default():
        freeze_var_names = \
            list(set(v.op.name for v in tf.compat.v1.global_variables()).difference(keep_var_names or []))
        logger.debug('Frozen graph has var names: %s', freeze_var_names)
        # tweaked here
        output_names = output_names or [out.op.name for out in keras_model.outputs]
        output_names += [v.op.name for v in tf.global_variables()]
        logger.debug('Frozen graph has output names: %s', output_names)
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ''

        try:
            frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(session, input_graph_def,
                                                                                  output_names, freeze_var_names)
        except Exception as e:
            logger.exception('Caught exception while converting variables to constants: %s', e)
            return -1

        logger.debug('Saving frozen graph to %s...', output_filepath)
        try:
            tf.io.write_graph(frozen_graph, os.path.dirname(output_filepath), os.path.split(output_filepath)[1],
                              as_text=False)
        except Exception as e:
            logger.exception('Caught exception while saving frozen graph: %s', e)
            return -1

    # Clean up as best we can
    logger.warn('Clearing keras session...')
    keras.backend.clear_session()
    logger.info('Deleting keras model...')
    del keras_model

    logger.info('Successfully froze graph to %s.', output_filepath)
    return 0


def convert_tf_lite(input_filepath, output_filepath, representative_data):
    assert input_filepath[-len(file_utils.EXTENSION_H5):] == file_utils.EXTENSION_H5, \
        'Unexpected extension for input file: %s' % input_filepath
    assert output_filepath[-len(file_utils.EXTENSION_INT8_TFLITE):] == file_utils.EXTENSION_INT8_TFLITE, \
        'Unexpected extension for output file: %s' % output_filepath
    assert representative_data is not None

    # seems to be required or we get errors with BatchNormalization
    logger.warn('Creating TFLiteConverter from %s...', input_filepath)
    if not os.path.exists(input_filepath):
        logger.error('File not found: %s', input_filepath)
        return -1

    # Below call will:
    # clear session
    # set learning phase False
    # loads model, gets sess.graph --> freezes --> __init__
    # TODO this is a useful function to trace
    # Thi sends up just getting a graphdef, input names, output names, and calling init.
    # but it restarts the session because it needs to do the freeze. So maybe we need to go to TF mode first?
    # That means that we should be able to implement this save by converting to tensorflow mode and then
    converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(input_filepath)

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

    logger.debug('Converting to tflite INT8 model...')
    try:
        tflite_model = converter.convert()
    except Exception as e:
        logger.exception('Caught exception while converting model to INT8 tflite: %s', e)
        return -1
    logger.debug('Saving tflite model to %s...', output_filepath)
    with open(output_filepath, 'wb') as o_:
        o_.write(tflite_model)

    # Clean up as best we can
    logger.warn('Clearing keras session...')
    keras.backend.clear_session()

    return 0





