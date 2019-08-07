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

MODELS_DIR = '../models/'


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
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=INPUT_SHAPE),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(OUTPUT_LEN, activation=tf.nn.softmax)
    ])
    model_compile(model)
    return model


def model_conv(hidden_size=2):
    # UpSampling2D (using nearest neighber) is not supported yet.
    # We should implement it by myself.
    def UpSampling2D(scale=(2, 2)):
        if isinstance(scale, int):
            scale = (scale, scale)

        def upsampling(x):
            shape = x.shape
            x = keras.layers.Concatenate(-2)([x] * scale[0])
            x = keras.layers.Reshape([shape[1] * scale[0], shape[2], shape[3]])(x)
            x = keras.layers.Concatenate(-1)([x] * scale[1])
            x = keras.layers.Reshape([shape[1] * scale[0], shape[2] * scale[1], shape[3]])(x)
            return x

        return upsampling

    inputs = keras.layers.Input(shape=INPUT_SHAPE)

    x = keras.layers.Conv2D(16, (3, 3), padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = keras.layers.Conv2D(8, (3, 3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = keras.layers.Conv2D(8, (3, 3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(OUTPUT_LEN, activation='softmax', name='fc%d' % OUTPUT_LEN)(x)
    model = keras.models.Model(inputs, x)
    model.summary()
    model_compile(model)

    return model


    encoded = keras.layers.Dense(hidden_size)(x)

    x = keras.layers.Dense(7 * 7 * 8)(encoded)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)


    x = keras.layers.Reshape((7, 7, 8))(x)
    x = UpSampling2D()(x)
    # x = keras.layers.UpSampling2D(2, interpolation='bilinear')(x)
    x = keras.layers.Conv2D(8, (3, 3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = UpSampling2D()(x)
    # x = keras.layers.UpSampling2D(2, interpolation='bilinear')(x)
    x = keras.layers.Conv2D(16, (3, 3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(100, (28, 28), strides=(28, 28), padding='same')(x)

    model = keras.models.Model(inputs, x)
    model.summary()
    model_compile(model)

    return model


def model_resnet():
    model = keras.models.Sequential()

    resnet50 = keras.applications.resnet50.ResNet50(include_top=False, input_shape=INPUT_SHAPE, pooling='avg')
    resnet50.summary()
    model.add(resnet50)

    model.add(keras.layers.Dense(OUTPUT_LEN, activation='softmax', name='fc%d' % OUTPUT_LEN))
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
            return model_resnet()
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
