#!/usr/bin/env python

import os
from src.utils.file_utils import TMP_DIR

UNKNOWN_EPOCH = -1

EXTENSION_H5 = '.h5'                             # architecture, weights, and optimizer
EXTENSION_ARCH_WEIGHTS_H5 = '_no_optimizer.h5'   # architecture, weights
EXTENSION_WEIGHTS_H5 = '_weights.h5'             # weights
EXTENSION_PB = '.pb'
EXTENSION_INT8_TFLITE = '_int8.tflite'

TENSORBOARD_DIR = os.path.join(TMP_DIR, 'tensorboard')
