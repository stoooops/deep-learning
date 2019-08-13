#!/usr/bin/env python

import tensorflow as tf
from tensorflow.python.client import device_lib


def devices():
    return device_lib.list_local_devices()


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def is_gpu_available():
    return tf.test.is_gpu_available()


def gpu_device_name():
    return tf.test.gpu_device_name()
