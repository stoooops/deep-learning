#!/usr/bin/env python


from tensorflow import keras

from src.meta_model import MetaModel


NAME = 'conv'


def construct_conv_model(input_shape, output_length):
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
