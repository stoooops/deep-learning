#!/usr/bin/env python

import os

EXTENSION_H5 = '.h5'                             # architecture, weights, and optimizer
EXTENSION_ARCH_WEIGHTS_H5 = '_no_optimizer.h5'   # architecture, weights
EXTENSION_WEIGHTS_H5 = '_weights.h5'             # weights
EXTENSION_PB = '.pb'
EXTENSION_INT8_TFLITE = '_int8.tflite'


FILE_DIR = os.path.dirname(os.path.realpath(__file__))
_EXPECTED_END_FILE_DIR = 'src/utils'
assert FILE_DIR[-len(_EXPECTED_END_FILE_DIR):] == _EXPECTED_END_FILE_DIR

SRC_DIR = os.path.normpath(os.path.join(FILE_DIR, '..'))

MODELS_DIR = os.path.normpath(os.path.join(SRC_DIR, '../models'))

TMP_DIR = os.path.normpath(os.path.join(SRC_DIR, '../tmp'))

LOG_DIR = os.path.normpath(os.path.join(TMP_DIR, 'logs'))


# Util functions for naming model files

# Directory

def model_dir(name, epoch):
    result = os.path.join(MODELS_DIR, model_filename_no_ext(name, epoch))
    if not os.path.exists(result):
        os.makedirs(result)
    return result


# Filename

def model_filename_no_ext(name, epoch):
    return '%s_%03d' % (name, epoch)


# .h5 - architecture/weights/optimizer

def model_filename_h5(name, epoch):
    return '%s%s' % (model_filename_no_ext(name, epoch), EXTENSION_H5)


def model_filepath_h5(name, epoch, dir_=None):
    return os.path.join(dir_ or model_dir(name, epoch), model_filename_h5(name, epoch))


# .h5 - architecture/weights

def model_filename_no_opt_h5(name, epoch):
    return '%s%s' % (model_filename_no_ext(name, epoch), EXTENSION_ARCH_WEIGHTS_H5)


def model_filepath_no_opt_h5(name, epoch, dir_=None):
    return os.path.join(dir_ or model_dir(name, epoch), model_filename_no_opt_h5(name, epoch))


# .h5 - weights

def model_filename_weights_h5(name, epoch):
    return '%s%s' % (model_filename_no_ext(name, epoch), EXTENSION_WEIGHTS_H5)


def model_filepath_weights_h5(name, epoch, dir_=None):
    return os.path.join(dir_ or model_dir(name, epoch), model_filename_weights_h5(name, epoch))


# .pb

def model_filename_pb(name, epoch):
    return '%s%s' % (model_filename_no_ext(name, epoch), EXTENSION_PB)


def model_filepath_pb(name, epoch, dir_=None):
    return os.path.join(dir_ or model_dir(name, epoch), model_filename_pb(name, epoch))


# .tflite - INT8

def model_filename_tflite(name, epoch):
    return '%s%s' % (model_filename_no_ext(name, epoch), EXTENSION_INT8_TFLITE)


def model_filepath_tflite(name, epoch, dir_=None):
    return os.path.join(dir_ or model_dir(name, epoch), model_filename_tflite(name, epoch))
