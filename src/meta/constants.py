#!/usr/bin/env python

import os
from src.utils.file_utils import LOG_DIR

UNKNOWN_EPOCH = -1

EXTENSION_H5 = '.h5'                             # architecture, weights, and optimizer
EXTENSION_ARCH_WEIGHTS_H5 = '_no_optimizer.h5'   # architecture, weights
EXTENSION_WEIGHTS_H5 = '_weights.h5'             # weights
EXTENSION_PB = '.pb'
EXTENSION_INT8_TFLITE = '_int8.tflite'

TENSORBOARD_LOG_DIR = os.path.join(LOG_DIR, 'tensorboard')
