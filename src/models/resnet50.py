#!/usr/bin/env python


from tensorflow import keras

from src.meta.keras import KerasModel
from src.meta.meta import MetaModel


NAME = 'resnet50'


def construct_resnet50_model(input_shape, output_length):
    keras_model = keras.models.Sequential(name=NAME)

    resnet50 = keras.applications.resnet50.ResNet50(include_top=False, input_shape=input_shape, pooling='avg')
    keras_model.add(resnet50)
    keras_model.add(keras.layers.Dense(output_length, activation='softmax', name='fc%d' % output_length))

    return keras_model
