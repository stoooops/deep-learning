#!/usr/bin/env python

import tensorflow as tf
from tensorflow import keras

from src.meta_model import MetaModel


NAME = 'basic'


def construct_basic_model(input_shape, output_length):
    keras_model = keras.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(output_length, activation=tf.nn.softmax)
    ])
    return MetaModel(NAME, keras_model=keras_model)
