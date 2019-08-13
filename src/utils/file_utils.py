#!/usr/bin/env python

import os
import datetime


##################################################
# File extensions
##################################################

EXTENSION_H5 = '.h5'                             # architecture, weights, and optimizer
EXTENSION_H5_ARCH_WEIGHTS = '_no_optimizer.h5'   # architecture, weights
EXTENSION_H5_WEIGHTS = '_weights.h5'             # weights
EXTENSION_MD = '.md'                             # metadata
EXTENSION_PB = '.pb'                             # protobuf
EXTENSION_INT8_TFLITE = '_int8.tflite'           # INT8 tflite


##################################################
# Static Directories
##################################################


FILE_DIR = os.path.dirname(os.path.realpath(__file__))
_EXPECTED_END_FILE_DIR = 'src/utils'
assert FILE_DIR[-len(_EXPECTED_END_FILE_DIR):] == _EXPECTED_END_FILE_DIR

SRC_DIR = os.path.normpath(os.path.join(FILE_DIR, '..'))

MODELS_DIR = os.path.normpath(os.path.join(SRC_DIR, '../models'))

TMP_DIR = os.path.normpath(os.path.join(SRC_DIR, '../tmp'))

LOG_DIR = os.path.normpath(os.path.join(TMP_DIR, 'logs'))

TENSORBOARD_LOG_DIR = os.path.join(LOG_DIR, 'tensorboard')


##################################################
# Model file/directory helper functions
##################################################

# Directories

def model_dir(name, epoch):
    # models/basic/001/
    result = os.path.join(os.path.join(MODELS_DIR, name), '%03d' % epoch)
    if not os.path.exists(result):
        os.makedirs(result)
    return result


def tensorboard_log_dir(name):
    # no need to eagerly create as tensorflow will auto-mkdir
    return os.path.join(os.path.join(TENSORBOARD_LOG_DIR, name),
                        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


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
    return '%s%s' % (model_filename_no_ext(name, epoch), EXTENSION_H5_ARCH_WEIGHTS)


def model_filepath_no_opt_h5(name, epoch, dir_=None):
    return os.path.join(dir_ or model_dir(name, epoch), model_filename_no_opt_h5(name, epoch))


# .h5 - weights

def model_filename_weights_h5(name, epoch):
    return '%s%s' % (model_filename_no_ext(name, epoch), EXTENSION_H5_WEIGHTS)


def model_filepath_weights_h5(name, epoch, dir_=None):
    return os.path.join(dir_ or model_dir(name, epoch), model_filename_weights_h5(name, epoch))


# .md - metadata

def model_filename_md(name, epoch):
    return '%s%s' % (model_filename_no_ext(name, epoch), EXTENSION_MD)


def model_filepath_md(name, epoch, dir_=None):
    return os.path.join(dir_ or model_dir(name, epoch), model_filename_md(name, epoch))


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
