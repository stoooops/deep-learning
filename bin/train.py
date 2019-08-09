#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import os
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
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
from src.models.basic import NAME as NAME_BASIC
from src.models.batchn import NAME as NAME_BATCHN
from src.models.conv import NAME as NAME_CONV
from src.models.resnet50 import NAME as NAME_RESNET50
from src.models import factory
from src.utils.file_utils import MODELS_DIR

logger = HuliLogging.get_logger(__name__)
HuliLogging.attach_stdout()

print('=' * 50)
print(tf.__name__, '-', tf.__version__, sep='')
print('=' * 50)


INPUT_SHAPE = (32, 32, 3)
OUTPUT_LEN = len(CIFAR_100_CLASSES)

MODEL_NAMES = [NAME_BASIC, NAME_BATCHN, NAME_CONV, NAME_RESNET50]


epoch = 0

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


def get_model(name, epoch):
    model_create_func = factory.get_model_create_func(name, INPUT_SHAPE, OUTPUT_LEN)
    ret, model = MetaModelFactory.from_weights_h5(name, epoch, model_create_func)
    if ret != 0:
        return ret, None
    model_compile(model)
    return 0, model


def evaluate(model, test_xy):
    now = time.time()
    test_loss, test_acc = model.evaluate(*test_xy)
    elapsed = time.time() - now
    print('[%.3fs] Test accuracy: %s' % (elapsed, test_acc))


def line():
    return '=' * 50


def log_prefix(epoch):
    return '[%s|%d]' % ('train', epoch)


def log_bold(*argv, **kwargs):
    global epoch
    logger.info(log_prefix(epoch) + ' ' + line() + ' ' + argv[0] + ' ' + line(), *argv[1:], **kwargs)


def train(model, train, test, epochs, initial_epoch=0, skip_pb=False, skip_tflite=False):
    """
    :type model: MetaModel
    """
    for prev_epoch in range(initial_epoch, epochs):
        next_epoch = prev_epoch + 1

        # Fit
        log_bold('FIT')
        model.fit(*train, epochs=next_epoch, initial_epoch=prev_epoch)
        global epoch
        epoch = next_epoch

        # Test
        log_bold('EVALUATE')
        evaluate(model, test)

        # Save
        recompile = False
        if not skip_pb and not skip_tflite:
            log_bold('SAVE_ALL')
            ret = model.save_all(representative_data=train[0])
            if ret != 0:
                logger.error('%s Failed saving all due to error %d', model.name, ret)
                exit(1)
            recompile = True
        else:
            # keras
            log_bold('SAVE')
            ret = model.save()
            if ret != 0:
                logger.error('%s Failed saving due to error %d', model.name, ret)
                exit(1)

            # pb
            if not skip_pb:
                log_bold('SAVE TO %s', TensorApi.TENSORFLOW)
                ret = model.save_to(TensorApi.TENSORFLOW)
                if ret != 0:
                    logger.error('%s Failed saving to %s due to error %d', model.name, TensorApi.TENSORFLOW, ret)
                    exit(1)
                recompile = True

            # tflite
            if not skip_tflite:
                log_bold('SAVE TO %s', TensorApi.TF_LITE)
                ret = model.save_to(TensorApi.TF_LITE, representative_data=train[0])
                if ret != 0:
                    logger.error('%s Failed saving to %s due to error %d', model.name, TensorApi.TF_LITE, ret)
                    exit(1)

        if recompile:
            # Recompile after saving, which can corrupt state because the model gets recreated
            log_bold('RECOMPILE')
            model_compile(model)




def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('-e', '--epochs', type=int, default=3, help='Training epochs.')
    p.add_argument('-i', '--initial-epoch', type=int, default=0, help='Initial epoch.')
    p.add_argument('-m', '--model', required=True, default=NAME_BASIC, help='Model name. One of %s' % MODEL_NAMES)
    p.add_argument('--skip-pb', default=False, help='Skip pb serialization', action="store_true")
    p.add_argument('--skip-tflite', default=False, help='Skip tflite serialization', action="store_true")
    args = p.parse_args()
    assert args.model in MODEL_NAMES
    assert args.epochs > args.initial_epoch
    return args


def main():
    # Args
    log_bold('PARSE')
    args = get_args()
    model_name = args.model
    initial_epoch = args.initial_epoch
    global epoch
    epoch = initial_epoch
    epochs = args.epochs
    skip_tflite = args.skip_tflite

    # Data
    log_bold('DATA')
    (train_images, train_labels), (test_images, test_labels) = get_data()

    # Model
    log_bold('MODEL')
    ret, model = get_model(model_name, initial_epoch)
    if ret != 0:
        logger.error('Getting model failed with error %d', ret)
        exit(1)
    model.dump()

    # Train
    log_bold('TRAIN')
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
