#!/usr/bin/env python

import tensorflow as tf
from tensorflow import keras

from src.meta_model import MetaModel


NAME = 'conv'


def construct_conv_model(input_shape, output_length):
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

    inputs = keras.layers.Input(shape=input_shape)

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
    x = keras.layers.Dense(output_length, activation='softmax', name='fc%d' % output_length)(x)
    keras_model = keras.models.Model(inputs, x)

    return MetaModel(NAME, keras_model=keras_model)
