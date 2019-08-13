#!/usr/bin/env python


import os
import pytest
from datetime import datetime

from tensorflow import keras


from src.data import cifar100
from src.meta.keras import KerasModel, _KerasModelSaver
from src.meta.metadata import Metadata
from src.meta.tensorflow import TensorFlowModel
from src.models import factory
from src.models.basic import NAME as BASIC
from src.models.batchn import NAME as BATCHN
from src.models.conv import NAME as CONV
from src.models.resnet50 import NAME as RESNET50
from src.utils.file_utils import EXTENSION_PB, EXTENSION_INT8_TFLITE, TMP_DIR
from src.utils.logger import Logging

Logging.attach_stdout()

DIR = os.path.join(TMP_DIR, datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(DIR)
print('Using tmp dir:', DIR)


(TRAIN_X, TRAIN_Y), (TEST_X, TEST_Y) = cifar100.load_data(normalize=True)
X_SHAPE = cifar100.INPUT_SHAPE
Y_LENGTH = 100


def f_basic():
    return factory.get_model_create_func(BASIC, X_SHAPE, Y_LENGTH)()


def f_batchn():
    return factory.get_model_create_func(BATCHN, X_SHAPE, Y_LENGTH)()


def f_conv():
    return factory.get_model_create_func(CONV, X_SHAPE, Y_LENGTH)()


def f_resnet50():
    return factory.get_model_create_func(RESNET50, X_SHAPE, Y_LENGTH)()


LIMIT = 3

F_KERAS_MODELS = [f_basic, f_batchn, f_conv, f_resnet50][:LIMIT]
NAMES = [BASIC, BATCHN, CONV, RESNET50][:LIMIT]
PARAMS = zip(NAMES, F_KERAS_MODELS)

# globals
_md = _km = None


def init_globals(metadata, f):
    assert metadata is not None and isinstance(metadata, Metadata)
    global _md, _km
    _md = metadata
    _km = KerasModel(_md.name, _md, f(), f)


def clear_session():
    keras.backend.clear_session()


def setup(name=None, f=None, epoch=0):
    if name is None or f is None:
        # This happens during pytest setup
        return
    clear_session()
    init_globals(Metadata(name, epoch=epoch), f)


def destroy():
    global _md, _km
    clear_session()
    _md = None
    del _km
    _km = None
    clear_session()


def assert_exists(filepath):
    assert os.path.exists(filepath), '%s does not exist' % filepath


@pytest.mark.parametrize("name,f_keras_model", PARAMS)
def test_keras_end_to_end(name, f_keras_model):
    setup(name, f_keras_model)
    global _md, _km

    ################################################################################################################
    # Compile
    ################################################################################################################

    # compile
    ret = _km.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    assert ret == 0, 'compile failed due to error %d' % name

    ################################################################################################################
    # Fit
    ################################################################################################################

    # fit
    ret, history = _km.fit(TRAIN_X, TRAIN_Y, epochs=1)
    assert ret == 0, '%s fit failed due to error %d' % (name, ret)
    assert _md.epoch == 1

    ################################################################################################################
    # Evaluate
    ################################################################################################################

    # evaluate
    ret, result = _km.evaluate(TEST_X, TEST_Y)
    assert ret == 0, '%s evaluate failed due to error %d' % (name, ret)

    ################################################################################################################
    # Predict
    ################################################################################################################

    # predict
    ret, y = _km.predict(TEST_X)
    assert ret == 0, '%s predict failed due to error %d' % (name, ret)

    ################################################################################################################
    # .h5
    ################################################################################################################

    # save
    filepath_h5 = _km.filepath_h5(dir_=DIR)
    assert not os.path.exists(filepath_h5)
    ret = _km.save(filepath_h5)
    assert ret == 0, '%s save failed due to error %d' % (name, ret)
    assert_exists(filepath_h5)

    # save weights
    filepath_weights_h5 = _km.filepath_weights_h5(dir_=DIR)
    assert not os.path.exists(filepath_weights_h5)
    ret = _km.save(filepath_weights_h5)
    assert ret == 0, '%s save failed due to error %d' % (name, ret)
    assert_exists(filepath_weights_h5)
    assert os.path.getsize(filepath_h5) >= os.path.getsize(filepath_weights_h5)

    # reload from function and weights
    before_input_names = _md.input_names
    before_output_names = _md.output_names
    ret = _km._reload_keras_model()
    assert ret == 0, '%s reload from function and weights failed due to error %d' % (name, ret)
    assert _md == _km.metadata
    assert before_input_names == _km.metadata.input_names
    assert before_output_names == _km.metadata.output_names

    # reload from h5
    before_input_names = _md.input_names
    before_output_names = _md.output_names
    ret = _km._reload_keras_model(force_h5=True)
    assert ret == 0, '%s reload from h5 failed due to error %d' % (name, ret)
    assert _md == _km.metadata
    assert before_input_names == _km.metadata.input_names
    assert before_output_names == _km.metadata.output_names

    ################################################################################################################
    # Restart Session
    ################################################################################################################

    # restart session
    before_input_names = _md.input_names
    before_output_names = _md.output_names
    ret = _km.restart_session(pre_save=False)
    assert ret == 0, '%s restart_session failed due to error %d' % (name, ret)
    assert _md == _km.metadata
    assert before_input_names == _km.metadata.input_names
    assert before_output_names == _km.metadata.output_names

    ################################################################################################################
    # .pb Freeze graph - freeze_graph()
    ################################################################################################################

    # freeze graph
    filepath_pb = _km.filepath_pb(dir_=DIR)
    assert not os.path.exists(filepath_pb)
    ret = _km.freeze_graph(filepath_pb)
    assert ret == 0, '%s freeze_session failed due to error %d' % (name, ret)
    assert_exists(filepath_pb)
    assert _md == _km.metadata

    # reload pb
    before_md = _md
    destroy()
    _md = before_md
    ret, tf_model = TensorFlowModel.load(name, _md, filepath_pb)
    assert ret == 0, '%s reload from pb failed due to error %d' % (name, ret)

    ################################################################################################################
    # .pb Full reload
    ################################################################################################################

    setup(name, f_keras_model)
    assert _md == _km.metadata
    assert before_input_names == _km.metadata.input_names
    assert before_output_names == _km.metadata.output_names

    ################################################################################################################
    # .pb Freeze graph - save()
    ################################################################################################################

    # Repeat freeze graph but use save API
    filepath_pb2 = _km.filepath_pb(dir_=DIR).replace(EXTENSION_PB, '_test2' + EXTENSION_PB)
    assert not os.path.exists(filepath_pb2)
    ret = _km.save(filepath_pb2)
    assert ret == 0, '%s save failed due to error %d' % (name, ret)
    assert_exists(filepath_pb2)
    assert _md == _km.metadata

    # reload pb
    before_md = _md
    destroy()
    _md = before_md
    ret, tf_model = TensorFlowModel.load(name, _md, filepath_pb2)
    assert ret == 0, '%s reload from pb failed due to error %d' % (name, ret)

    ################################################################################################################
    # .pb Full reload
    ################################################################################################################

    setup(name, f_keras_model)
    assert _md == _km.metadata
    assert before_input_names == _km.metadata.input_names
    assert before_output_names == _km.metadata.output_names

    ################################################################################################################
    # .tflite - directly from session
    ################################################################################################################

    has_batch_norm_layers = [BATCHN, CONV, RESNET50]
    if name not in has_batch_norm_layers:
        filepath_tflite = _km.filepath_tflite(dir_=DIR)
        assert not os.path.exists(filepath_tflite)
        ret = _km.save(filepath_tflite, representative_data=TRAIN_X, use_h5=False)
        assert ret == 0, '%s saving to %s via session failed due to error %d' % (name, filepath_tflite, ret)
        assert_exists(filepath_tflite)
        assert _md == _km.metadata

    ################################################################################################################
    # .tflite - from h5
    ################################################################################################################

    filepath_tflite2 = _km.filepath_tflite(dir_=DIR).replace(EXTENSION_INT8_TFLITE, '_test2' + EXTENSION_INT8_TFLITE)
    assert not os.path.exists(filepath_tflite2)
    ret = _KerasModelSaver(model=_km, filepath_h5=filepath_h5, log_prefix=_km.log_prefix())\
        .save(filepath_tflite2, representative_data=TRAIN_X, use_h5=True)
    assert ret == 0, '%s saving to %s failed via h5 file %s due to error %d' \
                     % (name, filepath_tflite2, filepath_h5, ret)
    assert_exists(filepath_tflite2)
    assert _md == _km.metadata

    ################################################################################################################
    # .pb Full reload
    ################################################################################################################

    setup(name, f_keras_model)
    assert _md == _km.metadata
    assert before_input_names == _km.metadata.input_names
    assert before_output_names == _km.metadata.output_names






