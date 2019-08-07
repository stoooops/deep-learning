#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import os
import time
import argparse
import numpy as np
from src.logger import HuliLogging

from src.models.basic import construct_basic_model
from src.models.conv import construct_conv_model
from src.models.resnet50 import construct_resnet50_model
from src.utils import MODELS_DIR

logger = HuliLogging.get_logger(__name__)

print('=' * 30)
print('tensorflow-%s' % tf.__version__)
print('=' * 30)


INPUT_SHAPE = (32, 32, 3)
CLASS_NAMES = np.array([
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        'worm'
    ])
OUTPUT_LEN = len(CLASS_NAMES)

MODEL_BASIC = 'basic'
MODEL_CONV = 'conv'
MODEL_RESNET50 = 'resnet50'
MODEL_NAMES = [MODEL_BASIC, MODEL_CONV, MODEL_RESNET50]


def get_filepath(model_name, epoch):
    filepath = os.path.join(MODELS_DIR, '%s_%03d.h5' % (model_name, epoch))
    return filepath


def get_data():
    cifar100 = keras.datasets.cifar100

    (train_images, train_labels), (test_images, test_labels) = cifar100.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    assert INPUT_SHAPE == train_images.shape[1:]
    print('Train:', train_images.shape)
    print('Test: ', test_images.shape)
    print('classes:', len(CLASS_NAMES))
    return (train_images, train_labels), (test_images, test_labels)


def model_compile(model):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


def model_basic():
    model = construct_basic_model(INPUT_SHAPE, OUTPUT_LEN)
    model_compile(model)
    return model


def model_conv():
    model = construct_conv_model(INPUT_SHAPE, OUTPUT_LEN)
    model_compile(model)
    return model


def model_resnet50():
    model = construct_resnet50_model(INPUT_SHAPE, OUTPUT_LEN)
    model_compile(model)
    return model


def get_model(name, epoch):
    if epoch == 0:
        logger.info('Creating new %s model...', name)
        if name == MODEL_BASIC:
            return model_basic()
        elif name == MODEL_CONV:
            return model_conv()
        elif name == MODEL_RESNET50:
            return model_resnet50()
        else:
            assert name in MODEL_NAMES
    else:
        filepath = get_filepath(name, epoch)
        print('Loading %s...' % filepath)
        return keras.models.load_model(filepath)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('-e', '--epochs', type=int, default=1, help='Training epochs.')
    p.add_argument('-i', '--initial-epoch', type=int, default=0, help='Initial epoch.')
    p.add_argument('-m', '--model', required=True, help='Model name. One of %s' % MODEL_NAMES)
    args = p.parse_args()
    assert args.model in MODEL_NAMES
    assert args.epochs > args.initial_epoch
    return args


def evaluate(model, test_xy):
    now = time.time()
    test_loss, test_acc = model.evaluate(*test_xy)
    elapsed = time.time() - now
    print('[%.3fs] Test accuracy: %s' % (elapsed, test_acc))


def save(model_name, model, epoch):
    filepath = get_filepath(model_name, epoch)
    print('Saving %s...' % filepath)
    model.save(filepath)


def train(model_name, model, train, test, epochs, initial_epoch=0):
    for prev_epoch in range(initial_epoch, epochs):
        epoch = prev_epoch + 1
        # Fit
        model.fit(*train, epochs=epoch, initial_epoch=prev_epoch)

        # Test
        evaluate(model, test)

        # Save
        save(model_name, model, epoch)

def main():
    # Args
    args = get_args()
    model_name = args.model
    initial_epoch = args.initial_epoch
    epochs = args.epochs

    # Data
    (train_images, train_labels), (test_images, test_labels) = get_data()

    # Model
    model = get_model(args.model, initial_epoch)
    model.summary()

    # Train
    train(model_name, model, (train_images, train_labels), (test_images, test_labels), epochs,
          initial_epoch=initial_epoch)


if __name__ == '__main__':
    main()
