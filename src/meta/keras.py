#!/usr/bin/env python

import os

import tensorflow as tf
from tensorflow import keras

from datetime import datetime
from src.meta.constants import EXTENSION_H5, TENSORBOARD_DIR
from src.meta.errors import *
from src.meta.metadata import Metadata
from src.meta.tensor_apis import AbstractTensorModel, TensorApi
from src.utils.logger import HuliLogging

from src.utils.file_utils import MODELS_DIR

logger = HuliLogging.get_logger(__name__)


class KerasModel(AbstractTensorModel):

    def __init__(self, name, metadata, keras_model, f_construct_keras_model=None):
        """
        :type name: str
        :type metadata: Metadata
        :type keras_model: keras.Model
        :type epoch: int
        """
        super().__init__(name, metadata, mode=TensorApi.KERAS)

        # keras Model
        assert keras_model is not None and isinstance(keras_model, keras.Model)
        self.keras_model = keras_model


        input_names = [op.name for op in self.keras_model.inputs]
        if self.metadata.input_names is not None and input_names != self.metadata.input_names:
            logger.warn('%s Overriding input names from %s to %s', self.log_prefix(), self.metadata.input_names, input_names)
        else:
            logger.debug('%s Setting metadata input names to %s', self.log_prefix(), input_names)
        self.metadata.input_names = input_names

        output_names = [op.name for op in self.keras_model.outputs]
        if self.metadata.output_names is not None and output_names != self.metadata.output_names:
            logger.warn('%s Overriding output names from %s to %s', self.log_prefix(), self.metadata.output_names, output_names)
        else:
            logger.debug('%s Setting metadata out names to %s', self.log_prefix(), output_names)
        self.metadata.output_names = output_names

        # tensorboard callback
        tensor_board_log_dir = os.path.join(TENSORBOARD_DIR,
                                            datetime.now().strftime("%Y%m%d-%H%M%S") + '_' + self.name)
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

        self.metadata.epoch = epochs
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
        # TODO auto determine filepath from epoch
        assert len(argv) == 1
        filepath = argv[0]

        ret = 0
        try:
            self.keras_model.save(filepath, **kwargs)
        except Exception as e:
            logger.exception('Undocumented exception: %s', e)
            ret = ERROR_TF_META_CAUGHT_EXCEPTION

        return ret

    def save_weights(self, *argv, **kwargs):
        # TODO auto determine filepath from epoch
        assert len(argv) == 1
        filepath = argv[0]

        ret = 0
        try:
            self.keras_model.save_weights(filepath, **kwargs)
        except Exception as e:
            logger.exception('Undocumented exception: %s', e)
            ret = ERROR_TF_META_CAUGHT_EXCEPTION

        return ret

    def summary(self, *argv, **kwargs):
        # Set print function, if not already set
        kwargs['print_fn'] = kwargs.get('print_fn', logger.info)
        ret = 0
        try:
            self.keras_model.summary(*argv, **kwargs)
        except ValueError as e:
            logger.exception(e)
            ret = ERROR_TF_META_CAUGHT_EXCEPTION
        return ret

    def dump(self):
        ret = self.summary(print_fn=logger.debug)
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

    def _reload_keras_model(self):
        if self.f_construct_keras_model is not None:
            self.keras_model = self.f_construct_keras_model()

            filepath_weights_h5 = self.filepath_weights_h5(self.metadata.epoch)
            logger.debug('%s Loading keras model weights from %s...', self.log_prefix(), filepath_weights_h5)
            self.keras_model.load_weights(filepath_weights_h5)
        else:
            filepath_h5 = self.filepath_h5(self.metadata.epoch)
            ret, self.keras_model = KerasModel._load_keras_model(filepath_h5)
            if ret != 0:
                logger.error('%s Failed re-loading keras model from %s due to error %d', self.log_prefix(), filepath_h5, ret)
                return ret
        return 0


    def restart_session(self, learning_phase=None):
        """
        Restart underlying tf.session, storing model to disk and back
        """
        # Save model to disk
        logger.debug('%s Saving h5 file so we can restart and reload session...', self.log_prefix())
        filepath_h5 = self.filepath_h5(self.metadata.epoch)
        ret = self.save(filepath_h5)
        if ret != 0:
            logger.error('%s Failed saving keras model to %s due to error %d', self.log_prefix(), filepath_h5, ret)
            return ret
        # Save model weights to disk
        logger.debug('%s Saving weights h5 file so we can restart and reload session...', self.log_prefix())
        filepath_weights_h5 = self.filepath_weights_h5(self.metadata.epoch)
        ret = self.save_weights(filepath_weights_h5)
        if ret != 0:
            logger.error('%s Failed saving keras model weights to %s due to error %d', self.log_prefix(), filepath_weights_h5,
                         ret)
            return ret

        # Clear session
        logger.warn('%s Bug Workaround: Clearing session...', self.log_prefix())
        keras.backend.clear_session()

        # This must be set before we reload the model
        if learning_phase is not None:
            logger.warn('%s Bug Workaround: Setting learning phase to %d...', self.log_prefix(), learning_phase)
            keras.backend.set_learning_phase(learning_phase)

        # Reload model from disk
        ret = self._reload_keras_model()
        if ret != 0:
            logger.error('%s Failed re-loading keras model from %s due to error %d', self.log_prefix(), filepath_h5, ret)
            return ret

        return 0

    def freeze_session(self, keep_var_names=None, output_names=None, clear_devices=True):
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
        # Bug workaround. Refer https://github.com/tensorflow/tensorflow/issues/31331#issuecomment-518655879
        # We should be able to remove this at some point
        logger.info('%s Bug Workaround: Restarting session...', self.log_prefix())
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
                logger.exception(e)
                return ERROR_TF_META_CAUGHT_EXCEPTION

            logger.debug('%s Saving frozen graph to %s...', self.log_prefix(), self.filepath_pb(self.metadata.epoch))
            try:
                tf.io.write_graph(frozen_graph, self.file_dir(self.metadata.epoch),
                                  self.filename_pb(self.metadata.epoch), as_text=False)
            except Exception as e:
                logger.exception(e)
                return ERROR_TF_META_CAUGHT_EXCEPTION

        # Bug workaround. Refer https://github.com/tensorflow/tensorflow/issues/31331#issuecomment-518655879
        # We should be able to remove this at some point
        logger.info('%s Bug Workaround: Restarting session...', self.log_prefix())
        ret = self.restart_session()
        if ret != 0:
            return ret

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
