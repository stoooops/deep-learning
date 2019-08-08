#!/usr/bin/env python

import os

import numpy as np
import tensorflow as tf

from src.meta.constants import UNKNOWN_EPOCH
from src.meta.errors import *
from src.meta.keras import KerasModel
from src.meta.tensorflow import TensorFlowModel
from src.meta.tflite import TfLiteModel
from src.meta.tensor_apis import AbstractTensorModel, TensorApi
from src.utils.cuda_utils import gpu_info
from src.utils.logger import HuliLogging


logger = HuliLogging.get_logger(__name__)

SUPPORTED_MODES = [TensorApi.NONE, TensorApi.KERAS, TensorApi.TENSOR_FLOW, TensorApi.TF_LITE]


class MetaModel(AbstractTensorModel):

    def __init__(self, name, epoch=UNKNOWN_EPOCH, delegate=None):
        super().__init__(name)
        self.mode = TensorApi.NONE

        assert isinstance(epoch, int) and (epoch == UNKNOWN_EPOCH or epoch >= 0)
        self.epoch = epoch

        assert delegate is None or isinstance(delegate, AbstractTensorModel)
        self.delegate = None
        if delegate is not None:
            self.attach_delegate(delegate)

    def attach_delegate(self, delegate):
        """
        :type delegate: AbstractTensorModel
        """
        assert self.mode == TensorApi.NONE
        assert isinstance(delegate, AbstractTensorModel), 'Expected AbstractTensorModel but got: %s' % delegate

        if delegate.mode == TensorApi.KERAS:
            assert isinstance(delegate, KerasModel)
        elif delegate.mode == TensorApi.TENSOR_FLOW:
            assert isinstance(delegate, TensorFlowModel)
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
        assert self.mode == TensorApi.KERAS  # TODO improve state logic

        filepath = self.filepath_h5(self.epoch) if self.mode == TensorApi.KERAS else None
        logger.debug('%s Saving keras model to %s...', self.name, filepath)
        ret = self.delegate.save(filepath, **kwargs)
        if ret != 0:
            return ret

        if self.mode == TensorApi.KERAS:
            # save weights
            filepath_weights = self.filepath_weights_h5(self.epoch)
            logger.debug('%s Saving keras model weights to %s...', self.name, filepath_weights)
            ret = self.delegate.save_weights(filepath_weights, **kwargs)
            if ret != 0:
                return ret

            # unless explicitly passed in as include_optimizer=True, then also store without the optimizer info
            if not kwargs.get('include_optimizer', False):
                filepath_no_opt = self.filepath_no_opt_h5(self.epoch)
                logger.debug('%s Saving keras model without optimizer to %s...', self.name, filepath_no_opt)
                kwargs['include_optimizer'] = True
                ret = self.delegate.save(filepath_no_opt, **kwargs)
                if ret != 0:
                    return ret

        return ret

    def dump(self):
        gpu_info()

        logger.debug('%s mode = %s', self.name, self.mode)
        logger.debug('%s epoch = %s', self.name, self.epoch)

        ret = 0
        if self.mode != TensorApi.NONE:
            ret = self.delegate.dump()
        return ret

    # New APIs

    def summary(self, *args, **kwargs):
        if self.mode != TensorApi.KERAS:
            logger.error('%s Summary not available in mode %s', self.name, self.mode)
            return ERROR_TF_META_WRONG_MODE
        return self.delegate.summary(*args, **kwargs)

    def reload(self):
        assert self.mode == TensorApi.KERAS
        before = self.mode

        logger.debug('%s Deleting existing delegate...', self.name)
        del self.delegate
        self.delegate = None
        self.mode = TensorApi.NONE

        if before == TensorApi.KERAS:
            ret, keras_model = KerasModel.load(self.name, self.epoch, self.filepath_h5(self.epoch))
            if ret != 0:
                return ERROR_TF_META_FAILED_RELOAD
            self.delegate = keras_model
        else:
            return ERROR_TF_META_UNIMPLEMENTED

        self.mode = before

        return 0

    def init_keras(self):
        ret, keras_model = KerasModel.load(self.name, self.epoch, self.filepath_h5(self.epoch))
        if ret != 0:
            return ret

        self.attach_delegate(keras_model)
        return 0

    def init_tflite(self):
        ret, tflite_model = TfLiteModel.load(self.name, self.epoch, self.filepath_tflite(self.epoch))
        if ret != 0:
            return ret

        self.attach_delegate(tflite_model)
        return 0

    def init_pb(self):
        ret, tf_model = TensorFlowModel.load(self.name, self.epoch, self.filepath_pb(self.epoch))
        if ret != 0:
            return ret

        self.attach_delegate(tf_model)
        return 0

    def change_mode(self, mode):
        if self.mode == mode:
            return 0

        if mode == TensorApi.NONE:
            del self.delegate
            self.delegate = None
            self.mode = mode
            return 0

        ret = self.change_mode(TensorApi.NONE)
        if ret != 0:
            return ret

        if mode == TensorApi.KERAS:
            ret = self.init_keras()
        elif mode == TensorApi.TENSOR_FLOW:
            ret = self.init_pb()
        elif mode == TensorApi.TF_LITE:
            ret = self.init_tflite()
        else:
            ret = ERROR_TF_META_BAD_INPUT

        return ret


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

        logger.debug('%s Saving mode %s via mode %s...', self.name, mode, self.mode)

        if self.mode == TensorApi.KERAS:
            if mode == TensorApi.TENSOR_FLOW:
                return 0
                return MetaModelModeConverter(self).save_pb()
            elif mode == TensorApi.TF_LITE:
                return MetaModelModeConverter(self).save_tflite(representative_data)
            else:
                assert 1 == 0, 'Unexpected mode: %s' % mode

    def save_all(self, representative_data=None):
        # Save keras file first
        save_order = [TensorApi.KERAS] + [t for t in SUPPORTED_MODES if t is not TensorApi.KERAS]
        for mode in save_order:
            if mode == TensorApi.NONE:
                continue
            if mode == self.mode:
                ret = self.save()
            elif mode == TensorApi.TENSOR_FLOW:
                ret = self.save_to(mode)
            elif mode == TensorApi.TF_LITE:
                ret = self.save_to(mode, representative_data=representative_data)
            else:
                ret = ERROR_TF_META_WRONG_MODE
            if ret != 0:
                return ret
        return 0


class MetaModelFactory:

    @staticmethod
    def from_h5(name, epoch):
        model = MetaModel(name, epoch=epoch)
        ret = model.init_keras()
        if ret != 0:
            return ret, None
        return 0, model

    @staticmethod
    def from_pb(name, epoch):
        model = MetaModel(name, epoch=epoch)
        ret = model.init_pb()
        if ret != 0:
            return ret, None
        return 0, model


    @staticmethod
    def from_tflite(name, epoch):
        model = MetaModel(name, epoch=epoch)
        ret = model.init_tflite()
        if ret != 0:
            return ret, None
        return 0, model


class MetaModelModeConverter:

    def __init__(self, meta_model):
        assert isinstance(meta_model, MetaModel)
        self.meta_model = meta_model

    def save_pb(self):
        return self.meta_model.freeze_session()

    def save_tflite(self, representative_data):
        h5_filepath = self.meta_model.filepath_h5(self.meta_model.epoch)
        if not os.path.exists(h5_filepath):
            logger.error('File not found: %s', h5_filepath)
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