#!/usr/bin/env python


from src.models.basic import construct_basic_model, NAME as BASIC
from src.models.batchn import construct_batchn_model, NAME as BATCHN
from src.models.conv import construct_conv_model, NAME as CONV
from src.models.resnet50 import construct_resnet50_model, NAME as RESNET50


def get_model_create_func(model_name, input_shape, output_length):
    if model_name == BASIC:
        return lambda: construct_basic_model(input_shape, output_length)
    elif model_name == BATCHN:
        return lambda: construct_batchn_model(input_shape, output_length)
    elif model_name == CONV:
        return lambda: construct_conv_model(input_shape, output_length)
    elif model_name == RESNET50:
        return lambda: construct_resnet50_model(input_shape, output_length)
    else:
        assert 0 == 1, 'Bad input'
