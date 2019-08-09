#!/usr/bin/env python

import tensorflow as tf
from tensorflow import keras

from src.meta.keras import KerasModel
from src.meta.metadata import Metadata
from src.meta.meta import MetaModel


NAME = 'basic'


def construct_basic_model(input_shape, output_length):
    inputs = keras.layers.Input(shape=input_shape, name=NAME+str(0))
    x = keras.layers.Flatten(input_shape=input_shape, name=NAME+str(1))(inputs)
    x = keras.layers.Dense(output_length, activation=tf.nn.softmax, name=NAME+str(2))(x)
    keras_model = keras.models.Model(inputs, x, name=NAME)

    return keras_model
