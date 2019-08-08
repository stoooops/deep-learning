#!/usr/bin/env python


from tensorflow import keras

from src.meta.keras import KerasModel
from src.meta.meta import MetaModel


NAME = 'batchn'


def construct_batchn_model(input_shape, output_length):
    inputs = keras.layers.Input(shape=input_shape, name=NAME+str(0))

    x = keras.layers.BatchNormalization()(inputs)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(output_length, activation='softmax', name='fc%d' % output_length)(x)
    keras_model = keras.models.Model(inputs, x, name=NAME)

    keras_model = KerasModel(NAME, keras_model, epoch=0)
    return MetaModel(NAME, epoch=0, delegate=keras_model)
