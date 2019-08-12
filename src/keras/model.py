#!/usr/bin/env python

import os

from tensorflow import keras

from datetime import datetime
from src.meta.constants import TENSORBOARD_LOG_DIR
from src.meta.errors import *
from src.meta.metadata import Metadata
from src.utils.logger import Logging
from src.utils import file_utils
from src.utils import io_utils

logger = Logging.get_logger(__name__)


class KerasModel:

    def __init__(self, name, metadata, keras_model, f_construct_keras_model=None):
        """
        :type name: str
        :type metadata: Metadata
        :type keras_model: keras.Model
        """
        assert name is not None and isinstance(name, str)
        self.name = name
        assert metadata is not None and isinstance(metadata, Metadata)
        self.metadata = metadata

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
        tensor_board_log_dir = os.path.join(TENSORBOARD_LOG_DIR,
                                            datetime.now().strftime("%Y%m%d-%H%M%S") + '_' + self.name)
        self.keras_tensorboard_callback = keras.callbacks.TensorBoard(log_dir=tensor_board_log_dir)

        # checkpoint
        file_dir = os.path.join(file_utils.MODELS_DIR, '%s_{epoch:03d}' % self.name)
        checkpoint_format = '%s_{epoch:03d}_checkpoint%s' % (self.name, file_utils.EXTENSION_H5)
        checkpoint_filepath_format = os.path.join(file_dir, checkpoint_format)
        self.keras_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath_format)

        self.f_construct_keras_model = f_construct_keras_model

    @staticmethod
    def from_weights_h5(name, epoch, f_construct_keras_model, filepath_weights_h5=None):
        metadata = Metadata(name, epoch=epoch)

        logger.debug('[%s|%d] Constructing keras model from factory function...', name, epoch)
        keras_model = f_construct_keras_model()

        filepath_weights_h5 = metadata.filepath_weights_h5() if filepath_weights_h5 is None else filepath_weights_h5
        logger.debug('[%s|%d] Loading keras model weights from %s...', name, epoch, filepath_weights_h5)
        keras_model.load_weights(filepath_weights_h5)

        model = KerasModel(name, metadata, keras_model, f_construct_keras_model=f_construct_keras_model)
        return 0, model

    # file dir

    def file_dir(self, epoch=None):
        epoch = self.metadata.epoch if epoch is None else epoch
        result = file_utils.model_dir(self.name, epoch)
        if not os.path.exists(result):
            os.makedirs(result)
        return result

    # .h5 - architecture/weights/optimizer

    def filename_h5(self):
        return self.metadata.filename_h5()

    def filepath_h5(self, dir_=None):
        return self.metadata.filepath_h5(dir_=dir_)

    # .h5 - architecture/weights

    def filename_no_opt_h5(self):
        return self.metadata.filename_no_opt_h5()

    def filepath_no_opt_h5(self, dir_=None):
        return self.metadata.filepath_no_opt_h5(dir_=dir_)

    # .h5 - weights

    def filename_weights_h5(self):
        return self.metadata.filename_weights_h5()

    def filepath_weights_h5(self, dir_=None):
        return self.metadata.filepath_weights_h5(dir_=dir_)

    # logging

    def log_prefix(self):
        return '[%s|%s|%s]' % (self.name, self.metadata.epoch, 'keras')

    # debug APIs

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

    # Keras APIs

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
            file_dir = self.file_dir(epoch=i)
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
        assert os.path.splitext(filepath)[1] == file_utils.EXTENSION_H5

        ret = 0
        try:
            self.keras_model.save(filepath, **kwargs)
        except Exception as e:
            logger.exception('Undocumented exception: %s', e)
            ret = ERROR_TF_META_CAUGHT_EXCEPTION

        return ret

    def save_weights(self, *argv, **kwargs):
        assert len(argv) <= 1
        filepath = argv[0] if len(argv) == 1 else self.filepath_weights_h5()
        assert os.path.splitext(filepath)[1] == file_utils.EXTENSION_H5

        ret = 0
        try:
            self.keras_model.save_weights(filepath, **kwargs)
        except Exception as e:
            logger.exception('Undocumented exception: %s', e)
            ret = ERROR_TF_META_CAUGHT_EXCEPTION

        return ret

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
