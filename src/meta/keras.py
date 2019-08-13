#!/usr/bin/env python

import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.meta.errors import *
from src.meta.metadata import Metadata
from src.meta.tensor_apis import AbstractTensorModel, AbstractTensorModelSaver, TensorApi
from src.utils.logger import Logging
from src.utils.file_utils import EXTENSION_H5, EXTENSION_PB, EXTENSION_INT8_TFLITE, MODELS_DIR
from src.utils import file_utils, io_utils

logger = Logging.get_logger(__name__)


class KerasModel(AbstractTensorModel):

    def __init__(self, name, metadata, keras_model, f_construct_keras_model=None):
        """
        :type name: str
        :type metadata: Metadata
        :type keras_model: keras.Model
        """
        super().__init__(name, metadata, mode=TensorApi.KERAS)

        # keras Model
        assert keras_model is not None and isinstance(keras_model, keras.Model),\
            'Expected keras.Model but got: %s' % keras_model
        self.keras_model = keras_model

        input_names = [op.name for op in self.keras_model.inputs]
        if self.metadata.input_names is not None and input_names != self.metadata.input_names:
            logger.warn('%s Overriding input names from %s to %s', self.log_prefix(), self.metadata.input_names,
                        input_names)
        else:
            logger.debug('%s Setting metadata input names to %s', self.log_prefix(), input_names)
        self.metadata.input_names = input_names

        output_names = [op.name for op in self.keras_model.outputs]
        if self.metadata.output_names is not None and output_names != self.metadata.output_names:
            logger.warn('%s Overriding output names from %s to %s', self.log_prefix(), self.metadata.output_names,
                        output_names)
        else:
            logger.debug('%s Setting metadata out names to %s', self.log_prefix(), output_names)
        self.metadata.output_names = output_names

        # tensorboard callback
        tensor_board_log_dir = file_utils.tensorboard_log_dir(self.name)
        self.keras_tensorboard_callback = keras.callbacks.TensorBoard(log_dir=tensor_board_log_dir)

        # checkpoint
        file_dir = os.path.join(MODELS_DIR, '%s_{epoch:03d}' % self.name)
        checkpoint_format = '%s_{epoch:03d}_checkpoint%s' % (self.name, EXTENSION_H5)
        checkpoint_filepath_format = os.path.join(file_dir, checkpoint_format)
        self.keras_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath_format)

        self.f_construct_keras_model = f_construct_keras_model

    def compile(self, *argv, **kwargs):
        try:
            self.keras_model.compile(*argv, **kwargs)
        except ValueError as e:
            logger.exception(e)
            return ERROR_TF_META_CAUGHT_EXCEPTION
        return 0

    def fit(self, *argv, **kwargs):
        assert self.keras_model is not None

        # attach checkpoint, tensorboard callback
        callbacks = [self.keras_checkpoint_callback, self.keras_tensorboard_callback]
        kwargs['callbacks'] = kwargs.get('callbacks', []) + callbacks
        logger.debug('%s keras tensorboard callback attached. Visualize by running: ', self.log_prefix())
        logger.debug('%s > tensorboard --logdir=%s', self.log_prefix(), self.keras_tensorboard_callback.log_dir)
        logger.debug('%s keras checkpoint callback attached. Logging to %s...', self.log_prefix(),
                     self.keras_checkpoint_callback.filepath)

        # ensure directories are init
        epochs = kwargs.get('epochs', 1)
        initial_epoch = kwargs.get('initial_epoch', 0)
        for i in range(initial_epoch + 1, epochs + 1):
            file_dir = self.file_dir(i)
            logger.info('%s Creating %s (if not already exists)...', self.log_prefix(), file_dir)

        # Call fit
        ret, history = 0, None
        try:
            history = self.keras_model.fit(*argv, **kwargs)
        except RuntimeError as e:
            logger.exception('%s Model was never compiled: %s', self.log_prefix(), e)
            ret = ERROR_TF_META_CAUGHT_EXCEPTION
        except ValueError as e:
            logger.exception('%s Mismatch between the provided input data and what the model expects: %s',
                             self.log_prefix(), e)
            ret = ERROR_TF_META_CAUGHT_EXCEPTION
        except Exception as e:
            logger.exception('%s Undocumented error: %s', self.log_prefix(), e)
            ret = ERROR_TF_META_CAUGHT_EXCEPTION
        if ret != 0:
            return ret, None

        self.metadata.update_epoch(epochs)
        return ret, history

    def evaluate(self, *argv, **kwargs):
        ret, result = 0, None
        try:
            result = self.keras_model.evaluate(*argv, **kwargs)
        except ValueError as e:
            logger.exception('Invalid arguments: %s', e)
            ret = ERROR_TF_META_CAUGHT_EXCEPTION
        except Exception as e:
            logger.exception('Undocumented error: %s', e)
            ret = ERROR_TF_META_CAUGHT_EXCEPTION
        return ret, result

    def predict(self, *argv, **kwargs):
        ret, y = 0, None
        try:
            y = self.keras_model.predict(*argv, **kwargs)
        except ValueError as e:
            logger.exception('Mismatch between the provided input data and the model\'s expectations, '
                             'or in case a stateful model receives a number of samples that is not a '
                             'multiple of the batch size: %s', e)
            ret = ERROR_TF_META_CAUGHT_EXCEPTION
        return ret, y

    def save(self, *argv, **kwargs):
        assert len(argv) <= 1
        filepath = argv[0] if len(argv) == 1 else self.filepath_h5()

        return _KerasModelSaver(model=self, log_prefix=self.log_prefix()).save(filepath, **kwargs)

    def save_weights(self, *argv, **kwargs):
        assert len(argv) <= 1
        filepath = argv[0] if len(argv) == 1 else self.filepath_weights_h5()

        return _KerasModelSaver(model=self, log_prefix=self.log_prefix()).save_weights(filepath, **kwargs)

    def summary(self, *argv, **kwargs):
        # Set print function, if not already set
        kwargs['print_fn'] = kwargs.get('print_fn', io_utils.prefix_print_fn(logger.debug, self.log_prefix()))
        ret = 0
        try:
            self.keras_model.summary(*argv, **kwargs)
        except ValueError as e:
            logger.exception(e)
            ret = ERROR_TF_META_CAUGHT_EXCEPTION
        return ret

    def dump(self):
        ret = self.summary(print_fn=io_utils.prefix_print_fn(logger.debug, self.log_prefix()))
        if ret != 0:
            return ret

        input_tensor = self.keras_model.input
        logger.debug('%s Input tensor: %s', self.log_prefix(), input_tensor)

        input_tensor_name = self.keras_model.input.name.split(':')[0]
        logger.debug('%s Input tensor name: %s', self.log_prefix(), input_tensor_name)

        output_tensor = self.keras_model.output
        logger.debug('%s Output tensor: %s', self.log_prefix(), output_tensor)

        output_tensor_name = self.keras_model.output.name.split(':')[0]
        logger.debug('%s Output tensor name: %s', self.log_prefix(), output_tensor_name)

        return 0

    def _reload_keras_model(self, force_h5=False):
        logger.debug('%s Reloaded keras model...', self.log_prefix())
        if not force_h5 and self.f_construct_keras_model is not None:
            self.keras_model = self.f_construct_keras_model()

            filepath_weights_h5 = self.filepath_weights_h5()
            logger.debug('%s Loading keras model weights from %s...', self.log_prefix(), filepath_weights_h5)
            self.keras_model.load_weights(filepath_weights_h5)
        else:
            filepath_h5 = self.filepath_h5()
            ret, self.keras_model = KerasModel._load_keras_model(filepath_h5)
            if ret != 0:
                logger.error('%s Failed re-loading keras model from %s due to error %d', self.log_prefix(), filepath_h5,
                             ret)
                return ret

        logger.debug('%s Successfully reloaded keras model.', self.log_prefix())
        return 0

    def restart_session(self, pre_save=True, learning_phase=None):
        """
        Restart underlying tf.session, storing model to disk and back
        """
        logger.debug('%s Restarting keras session...'), self.log_prefix()
        if pre_save:
            # Save model to disk
            logger.debug('%s Saving h5 file so we can restart and reload session...', self.log_prefix())
            filepath_h5 = self.filepath_h5()
            ret = self.save(filepath_h5)
            if ret != 0:
                logger.error('%s Failed saving keras model to %s due to error %d', self.log_prefix(), filepath_h5, ret)
                return ret
            # Save model weights to disk
            logger.debug('%s Saving weights h5 file so we can restart and reload session...', self.log_prefix())
            filepath_weights_h5 = self.filepath_weights_h5()
            ret = self.save_weights(filepath_weights_h5)
            if ret != 0:
                logger.error('%s Failed saving keras model weights to %s due to error %d', self.log_prefix(),
                             filepath_weights_h5, ret)
                return ret

        # Clear session
        logger.warn('%s Clearing session...', self.log_prefix())
        keras.backend.clear_session()

        # This must be set before we reload the model
        if learning_phase is not None:
            logger.warn('%s Bug Workaround: Setting learning phase to %d...', self.log_prefix(), learning_phase)
            keras.backend.set_learning_phase(learning_phase)

        # Reload model from disk (using factory function and weights if possible)
        logger.debug('%s Reloading keras model...', self.log_prefix())
        ret = self._reload_keras_model()
        if ret != 0:
            logger.error('%s Failed re-loading keras model from %s due to error %d', self.log_prefix(), filepath_h5,
                         ret)
            return ret

        logger.debug('%s Successfully restarted keras session.', self.log_prefix())
        return 0

    def freeze_graph(self, filepath, keep_var_names=None, output_names=None, clear_devices=True):
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
        assert os.path.splitext(filepath)[1] == EXTENSION_PB
        # Bug workaround. Refer https://github.com/tensorflow/tensorflow/issues/31331#issuecomment-518655879
        # We should be able to remove this at some point
        logger.info('%s Freezing graph...', self.log_prefix())
        logger.debug('%s Bug Workaround: Restarting session with learning phase=0...', self.log_prefix())
        ret = self.restart_session(learning_phase=0)
        if ret != 0:
            return ret
        # Now do freeze

        # Refer https://stackoverflow.com/a/45466355/2079993
        session = tf.compat.v1.keras.backend.get_session()
        graph = session.graph
        with graph.as_default():
            freeze_var_names =\
                list(set(v.op.name for v in tf.compat.v1.global_variables()).difference(keep_var_names or []))
            logger.debug('%s Frozen graph has var names: %s', self.log_prefix(), freeze_var_names)
            # tweaked here
            output_names = output_names or [out.op.name for out in self.keras_model.outputs]
            output_names += [v.op.name for v in tf.global_variables()]
            logger.debug('%s Frozen graph has output names: %s', self.log_prefix(), output_names)
            # Graph -> GraphDef ProtoBuf
            input_graph_def = graph.as_graph_def()
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ""

            try:
                frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(session, input_graph_def,
                                                                                      output_names, freeze_var_names)
            except Exception as e:
                logger.exception('Caught exception while converting variables to constants: %s', e)
                return ERROR_TF_META_CAUGHT_EXCEPTION

            logger.debug('%s Saving frozen graph to %s...', self.log_prefix(), filepath)
            try:
                tf.io.write_graph(frozen_graph, os.path.dirname(filepath), os.path.split(filepath)[1], as_text=False)
            except Exception as e:
                logger.exception('Caught exception while saving frozen graph: %s', e)
                return ERROR_TF_META_CAUGHT_EXCEPTION

        # Bug workaround. Refer https://github.com/tensorflow/tensorflow/issues/31331#issuecomment-518655879
        # We should be able to remove this at some point
        logger.debug('%s Bug Workaround: Restarting session...', self.log_prefix())
        ret = self.restart_session(pre_save=False)
        if ret != 0:
            return ret

        logger.info('%s Successfully froze graph to %s.', self.log_prefix(), filepath)
        return 0

    @staticmethod
    def _load_keras_model(filepath):
        assert filepath is not None and isinstance(filepath, str)
        try:
            keras_model = keras.models.load_model(filepath)
        except IOError as e:
            logger.exception(e)
            return ERROR_TF_META_CAUGHT_EXCEPTION, None
        except Exception as e:
            logger.exception(e)
            return ERROR_TF_META_CAUGHT_EXCEPTION, None

        return 0, keras_model

    @staticmethod
    def load(name, metadata, filepath):
        assert name is not None and isinstance(name, str)
        assert metadata is not None and isinstance(metadata, Metadata)
        assert filepath is not None and isinstance(filepath, str)

        ret, keras_model = KerasModel._load_keras_model(filepath)
        if ret != 0:
            return ret, None

        result = KerasModel(name, metadata, keras_model)

        return 0, result


class _KerasModelSaver(AbstractTensorModelSaver):
    """
    Wrapper for various saving functionality.

    These objects are short-lived and just served to compartmentalize the saving routines.
    """

    def __init__(self, model=None, filepath_h5=None, log_prefix=''):
        """
        :type model: KerasModel
        :type filepath_h5: str
        """
        assert model is None or isinstance(model, KerasModel)
        self._model = model

        # h5 file can optionally be used
        assert filepath_h5 is None or \
               (isinstance(filepath_h5, str) and os.path.splitext(filepath_h5)[1] == EXTENSION_H5),\
            'Bad h5 filepath: %s'
        self._model_h5 = filepath_h5
        if self._model_h5 is not None:
            assert os.path.exists(self._model_h5)

        self._log_prefix = log_prefix

    def save(self, filepath, representative_data=None, **kwargs):
        extension = os.path.splitext(filepath)[1]
        if extension == EXTENSION_H5:
            assert representative_data is None
            ret = self._save_model_h5(filepath, **kwargs)

        elif extension == EXTENSION_PB:
            assert representative_data is None
            ret = self._save_model_pb(filepath, **kwargs)

        elif filepath[-len(EXTENSION_INT8_TFLITE):] == EXTENSION_INT8_TFLITE:
            assert representative_data is not None
            ret = self._save_model_tflite(filepath, representative_data=representative_data, **kwargs)

        else:
            ret = ERROR_TF_META_BAD_INPUT
        return ret

    def _save_model_h5(self, filepath, **kwargs):
        assert os.path.splitext(filepath)[1] == EXTENSION_H5
        assert self._model is not None

        ret = 0
        try:
            self._model.keras_model.save(filepath, **kwargs)
        except Exception as e:
            logger.exception('Undocumented exception: %s', e)
            ret = ERROR_TF_META_CAUGHT_EXCEPTION

        return ret

    def _save_model_pb(self, filepath):
        assert os.path.splitext(filepath)[1] == EXTENSION_PB
        assert self._model is not None

        return self._model.freeze_graph(filepath)

    def _save_model_tflite(self, filepath, representative_data, use_h5=True):
        assert filepath[-len(EXTENSION_INT8_TFLITE):] == EXTENSION_INT8_TFLITE, 'Bad filepath: %s' % filepath
        assert use_h5 == (self._model_h5 is not None), 'use_h5=%s but self.model_hb is: %s' % (use_h5, self._model_h5)

        if use_h5:
            log_prefix = '%s %s' % (self._log_prefix, ' [BatchN workaround]')
            # seems to be required or we get errors with BatchNormalization
            logger.warn('%s Creating TFLiteConverter from %s...', log_prefix, self._model_h5)
            if not os.path.exists(self._model_h5):
                logger.error('File not found: %s', self._model_h5)
                return ERROR_TF_META_FILE_NOT_FOUND

            # Below call will:
            # clear session
            # set learning phase False
            # loads model, gets sess.graph --> freezes --> __init__
            # TODO this is a useful function to trace
            # Thi sends up just getting a graphdef, input names, output names, and calling init.
            # but it restarts the session because it needs to do the freeze. So maybe we need to go to TF mode first?
            # That means that we should be able to implement this save by converting to tensorflow mode and then
            converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(self._model_h5)
            # above call clears the keras session, so we need to reload our model
            if self._model.mode == TensorApi.KERAS:
                logger.info('%s Reloading since loading from keras model will clear session...', log_prefix)
                logger.debug('%s TODO can we do this better?', log_prefix)
                # TODO really I just need to minimally restore the session since the above call did the following:
                ret = self._model.restart_session(pre_save=False)
                if ret != 0:
                    return ret
        else:
            # This works for non-BatchNormalization
            # prefer this since if we use from_keras_model_file() then it clears the session
            logger.debug('%s Creating TFLiteConverter from existing keras session...', self._log_prefix)
            converter = tf.lite.TFLiteConverter.from_session(keras.backend.get_session(),
                                                             self._model.keras_model.inputs,
                                                             self._model.keras_model.outputs)

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

        logger.debug('%s Converting to tflite INT8 model...', self._log_prefix)
        try:
            tflite_model = converter.convert()
        except Exception as e:
            logger.exception('%s Caught exception while converting model to INT8 tflite: %s', self._log_prefix, e)
            return ERROR_TF_META_CAUGHT_EXCEPTION
        logger.debug('%s Saving tflite model to %s...', self._log_prefix, filepath)
        with open(filepath, 'wb') as o_:
            o_.write(tflite_model)

        return 0

    def save_weights(self, filepath, **kwargs):
        extension = os.path.splitext(filepath)[1]
        assert extension == EXTENSION_H5

        ret = 0
        try:
            self._model.keras_model.save_weights(filepath, **kwargs)
        except Exception as e:
            logger.exception('Undocumented exception: %s', e)
            ret = ERROR_TF_META_CAUGHT_EXCEPTION

        return ret
