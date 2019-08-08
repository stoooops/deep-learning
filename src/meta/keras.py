#!/usr/bin/env python

import os

import tensorflow as tf
from tensorflow import keras

from datetime import datetime
from src.meta.constants import EXTENSION_H5, TENSORBOARD_DIR, UNKNOWN_EPOCH
from src.meta.errors import *
from src.meta.tensor_apis import AbstractTensorModel, TensorApi
from src.utils.logger import HuliLogging

from src.utils.file_utils import MODELS_DIR

logger = HuliLogging.get_logger(__name__)


class KerasModel(AbstractTensorModel):

    def __init__(self, name, keras_model, epoch=UNKNOWN_EPOCH):
        """
        :type name: str
        :type keras_model: keras.Model
        :type epoch: int
        """
        super().__init__(name)

        assert isinstance(epoch, int) and (epoch == UNKNOWN_EPOCH or epoch >= 0)
        self.epoch = epoch

        # keras Model
        assert keras_model is not None and isinstance(keras_model, keras.Model)
        self.keras_model = keras_model

        # tensorboard callback
        tensor_board_log_dir = os.path.join(TENSORBOARD_DIR,
                                            datetime.now().strftime("%Y%m%d-%H%M%S") + '_' + self.name)
        self.keras_tensorboard_callback = keras.callbacks.TensorBoard(log_dir=tensor_board_log_dir)

        # checkpoint
        file_dir = os.path.join(MODELS_DIR, '%s_{epoch:03d}' % self.name)
        checkpoint_format = '%s_{epoch:03d}_checkpoint%s' % (self.name, EXTENSION_H5)
        checkpoint_filepath_format = os.path.join(file_dir, checkpoint_format)
        self.keras_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath_format)

        self.mode = TensorApi.KERAS

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
        logger.debug('%s keras tensorboard callback attached. Visualize by running: ', self.name)
        logger.debug('> tensorboard --logdir=%s', self.keras_tensorboard_callback.log_dir)
        logger.debug('%s keras checkpoint callback attached. Logging to %s...', self.name,
                     self.keras_checkpoint_callback.filepath)

        # ensure directories are init
        epochs = kwargs.get('epochs', 1)
        initial_epoch = kwargs.get('initial_epoch', 0)
        for i in range(initial_epoch + 1, epochs + 1):
            file_dir = self.file_dir(i)
            logger.info('Creating %s (if not already exists)...', file_dir)

        # Call fit
        ret, history = 0, None
        try:
            history = self.keras_model.fit(*argv, **kwargs)
        except RuntimeError as e:
            logger.exception('Model was never compiled: %s', e)
            ret = ERROR_TF_META_CAUGHT_EXCEPTION
        except ValueError as e:
            logger.exception('Mismatch between the provided input data and what the model expects: %s', e)
            ret = ERROR_TF_META_CAUGHT_EXCEPTION
        except Exception as e:
            logger.exception('Undocumented error: %s', e)
            ret = ERROR_TF_META_CAUGHT_EXCEPTION
        if ret != 0:
            return ret, None

        self.epoch = epochs
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
        logger.debug('%s Input tensor: %s', self.name, input_tensor)

        input_tensor_name = self.keras_model.input.name.split(':')[0]
        logger.debug('%s Input tensor name: %s', self.name, input_tensor_name)

        output_tensor = self.keras_model.output
        logger.debug('%s Output tensor: %s', self.name, output_tensor)

        output_tensor_name = self.keras_model.output.name.split(':')[0]
        logger.debug('%s Output tensor name: %s', self.name, output_tensor_name)

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

            try:
                frozen_graph = tf.graph_util.convert_variables_to_constants(session, input_graph_def, output_names,
                                                                            freeze_var_names)
            except Exception as e:
                logger.exception(e)
                return ERROR_TF_META_CAUGHT_EXCEPTION

            logger.info('%s Saving frozen graph to %s...', self.name, self.filepath_pb(self.epoch))
            try:
                tf.train.write_graph(frozen_graph, self.file_dir(), self.filename_pb(self.epoch), as_text=False)
            except Exception as e:
                logger.exception(e)
                return ERROR_TF_META_CAUGHT_EXCEPTION

        return 0

    @staticmethod
    def load(name, epoch, filepath):
        logger.debug('%s Loading keras model from %s...', name, filepath)
        try:
            keras_model = keras.models.load_model(filepath)
        except IOError as e:
            logger.exception(e)
            return ERROR_TF_META_CAUGHT_EXCEPTION, None
        except Exception as e:
            logger.exception(e)
            return ERROR_TF_META_CAUGHT_EXCEPTION, None

        result = KerasModel(name, keras_model, epoch=epoch)

        return 0, result
