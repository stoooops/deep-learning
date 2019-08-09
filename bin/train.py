#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import os
import sys
import time
import argparse
from src.utils.logger import HuliLogging

from src.data.cifar100 import CIFAR_100_CLASSES, load_cifar100_data
from src.meta.meta import MetaModel, MetaModelFactory
from src.meta.tensor_apis import TensorApi
from src.models.basic import construct_basic_model, NAME as NAME_BASIC
from src.models.batchn import construct_batchn_model, NAME as NAME_BATCHN
from src.models.conv import construct_conv_model, NAME as NAME_CONV
from src.models.resnet50 import construct_resnet50_model, NAME as NAME_RESNET50
from src.utils.file_utils import MODELS_DIR

logger = HuliLogging.get_logger(__name__)
HuliLogging.attach_stdout()

print('=' * 50)
print(tf.__name__, '-', tf.__version__, sep='')
print('=' * 50)


INPUT_SHAPE = (32, 32, 3)
OUTPUT_LEN = len(CIFAR_100_CLASSES)

MODEL_NAMES = [NAME_BASIC, NAME_BATCHN, NAME_CONV, NAME_RESNET50]


def get_filepath(model_name, epoch):
    filepath = os.path.join(MODELS_DIR, '%s_%03d.h5' % (model_name, epoch))
    return filepath


def get_data():
    (train_images, train_labels), (test_images, test_labels) = load_cifar100_data()
    assert INPUT_SHAPE == train_images.shape[1:]
    logger.info('Train: %s', train_images.shape)
    logger.info('Test: %s', test_images.shape)
    logger.info('Classes: %s', len(CIFAR_100_CLASSES))
    return (train_images, train_labels), (test_images, test_labels)


def model_compile(model):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


def model_basic():
    model = construct_basic_model(INPUT_SHAPE, OUTPUT_LEN)
    model_compile(model)
    return 0, model


def model_batchn():
    model = construct_batchn_model(INPUT_SHAPE, OUTPUT_LEN)
    model_compile(model)
    return 0, model


def model_conv():
    model = construct_conv_model(INPUT_SHAPE, OUTPUT_LEN)
    model_compile(model)
    return 0, model


def model_resnet50():
    model = construct_resnet50_model(INPUT_SHAPE, OUTPUT_LEN)
    model_compile(model)
    return 0, model


def get_model(name, epoch):
    if epoch == 0:
        logger.info('Creating new %s model...', name)
        if name == NAME_BASIC:
            return model_basic()
        if name == NAME_BATCHN:
            return model_batchn()
        elif name == NAME_CONV:
            return model_conv()
        elif name == NAME_RESNET50:
            return model_resnet50()
        else:
            assert name in MODEL_NAMES
    else:
        return MetaModelFactory.from_h5(name, epoch)


def evaluate(model, test_xy):
    now = time.time()
    test_loss, test_acc = model.evaluate(*test_xy)
    elapsed = time.time() - now
    print('[%.3fs] Test accuracy: %s' % (elapsed, test_acc))


def train(model, train, test, epochs, initial_epoch=0, skip_pb=False, skip_tflite=False):
    """
    :type model: MetaModel
    """
    for prev_epoch in range(initial_epoch, epochs):
        epoch = prev_epoch + 1
        # Fit
        model.fit(*train, epochs=epoch, initial_epoch=prev_epoch)

        # Test
        evaluate(model, test)

        # Save
        if not skip_pb and not skip_tflite:
            ret = model.save_all(representative_data=train[0])
            if ret != 0:
                logger.error('%s Failed saving all due to error %d', model.name, ret)
                exit(1)
        else:
            # keras
            ret = model.save()
            if ret != 0:
                logger.error('%s Failed saving due to error %d', model.name, ret)
                exit(1)

            # pb
            if not skip_pb:
                ret = model.save_to(TensorApi.TENSORFLOW)
                if ret != 0:
                    logger.error('%s Failed saving to %s due to error %d', model.name, TensorApi.TENSORFLOW, ret)
                    exit(1)

            # tflite
            if not skip_tflite:
                ret = model.save_to(TensorApi.TF_LITE, representative_data=train[0])
                if ret != 0:
                    logger.error('%s Failed saving to %s due to error %d', model.name, TensorApi.TF_LITE, ret)
                    exit(1)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('-e', '--epochs', type=int, default=1, help='Training epochs.')
    p.add_argument('-i', '--initial-epoch', type=int, default=0, help='Initial epoch.')
    p.add_argument('-m', '--model', required=True, help='Model name. One of %s' % MODEL_NAMES)
    p.add_argument('--skip-pb', default=False, help='Skip pb serialization', action="store_true")
    p.add_argument('--skip-tflite', default=False, help='Skip tflite serialization', action="store_true")
    args = p.parse_args()
    assert args.model in MODEL_NAMES
    assert args.epochs > args.initial_epoch
    return args


def main():
    # Args
    args = get_args()
    model_name = args.model
    initial_epoch = args.initial_epoch
    epochs = args.epochs
    skip_tflite = args.skip_tflite

    # Data
    (train_images, train_labels), (test_images, test_labels) = get_data()

    # Model
    ret, model = get_model(model_name, initial_epoch)
    if ret != 0:
        logger.error('Getting model failed with error %d', ret)
        exit(1)
    model.dump()

    # Train
    train(model, (train_images, train_labels), (test_images, test_labels), epochs, initial_epoch=initial_epoch,
          skip_tflite=skip_tflite)


if __name__ == '__main__':
    now = time.time()
    logger.info('')
    logger.info('')
    logger.info('> ' + ' '.join(sys.argv))
    try:
        main()
    except Exception as e:
        logger.exception(e)
        exit(1)

    logger.info('')
    logger.info('> ' + ' '.join(sys.argv))
    logger.info('')
    logger.info('[%.3fs] SUCCESS!!!', time.time() - now)
