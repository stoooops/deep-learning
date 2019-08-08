#!/usr/bin/env python

import tensorflow as tf
from tensorflow import keras

from src.meta.keras import KerasModel
from src.meta.meta import MetaModel


NAME = 'basic'


def construct_basic_model(input_shape, output_length):
    keras_model = keras.Sequential(name=NAME, layers=[
        keras.layers.Flatten(input_shape=input_shape, name=NAME+str(0)),
        keras.layers.Dense(128, activation=tf.nn.relu, name=NAME+str(1)),
        keras.layers.Dense(output_length, activation=tf.nn.softmax, name=NAME+str(2))
    ])
    keras_model = KerasModel(NAME, keras_model, epoch=0)
    return MetaModel(NAME, epoch=0, delegate=keras_model)
