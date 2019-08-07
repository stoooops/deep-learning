#!/usr/bin/env python

import tensorflow as tf
from tensorflow import keras

from src.meta_model import MetaModel


NAME = 'resnet50'


def construct_resnet50_model(input_shape, output_length):
    keras_model = keras.models.Sequential()

    resnet50 = keras.applications.resnet50.ResNet50(include_top=False, input_shape=input_shape, pooling='avg')
    keras_model.add(resnet50)
    keras_model.add(keras.layers.Dense(output_length, activation='softmax', name='fc%d' % output_length))

    return MetaModel(NAME, keras_model=keras_model)
