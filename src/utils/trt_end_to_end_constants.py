#!/usr/bin/env python

import os
import json
import jsonpickle


# custom md
MD_FILE_FORMAT = '%s_epoch%03d_%s.md'

# keras h5
H5_FILE_FORMAT = '%s_epoch%03d_%s.h5'

# tf saved graph
SAVED_GRAPH_FILE_FORMAT = 'tf_model_%s_epoch%03d_%s'
SAVED_GRAPH_META_FILE_FORMAT = 'tf_model_%s_epoch%03d_%s.meta'

# tf frozen graph
FROZEN_GRAPH_FILE_FORMAT = '%s_epoch%03d_%s_frozen.pb'

# TRT
TRT_GRAPH_FILE_FORMAT = '%s_epoch%03d_%s_trt.pb'

# Dirs
_NAME_EPOCH_DIR_FORMAT = '%s/%03d'
_TMP_DIR = '../../../tmp'
_TRT_END_TO_END_DIR = os.path.join(_TMP_DIR, 'trt_end_to_end')

_TRAIN_DIR = os.path.join(_TRT_END_TO_END_DIR, 'train')


def get_train_dir(name, epoch):
    result = os.path.join(_TRAIN_DIR, _NAME_EPOCH_DIR_FORMAT % (name, epoch))
    if not os.path.exists(result):
        os.makedirs(result)
    return result


_CONVERT_DIR = os.path.join(_TRT_END_TO_END_DIR, 'convert')

_CONVERT_TF_DIR = os.path.join(_CONVERT_DIR, 'tf')
_CONVERT_TF_SAVED_GRAPH_DIR = os.path.join(_CONVERT_TF_DIR, 'saved')


def get_saved_graph_dir(name, epoch):
    result = os.path.join(_CONVERT_TF_SAVED_GRAPH_DIR, _NAME_EPOCH_DIR_FORMAT % (name, epoch))
    if not os.path.exists(result):
        os.makedirs(result)
    return result


_CONVERT_TF_FROZEN_GRAPH_DIR = os.path.join(_CONVERT_TF_DIR, 'frozen')


def get_frozen_graph_dir(name, epoch):
    result = os.path.join(_CONVERT_TF_FROZEN_GRAPH_DIR, _NAME_EPOCH_DIR_FORMAT % (name, epoch))
    if not os.path.exists(result):
        os.makedirs(result)
    return result


_CONVERT_TRT_FROZEN_GRAPH_DIR = os.path.join(_CONVERT_DIR, 'trt')


def get_trt_graph_dir(name, epoch):
    result = os.path.join(_CONVERT_TRT_FROZEN_GRAPH_DIR, _NAME_EPOCH_DIR_FORMAT % (name, epoch))
    if not os.path.exists(result):
        os.makedirs(result)
    return result


def save_params(name_, epoch_, time_):
    save_to_dir = get_train_dir(name_, epoch_)
    params = {'name': name_, 'epoch': epoch_, 'time': time_}
    params_filepath = os.path.join(save_to_dir, 'params.json')

    if os.path.exists(save_to_dir):
        print('Deleting', save_to_dir, '...', sep='')
        import shutil
        shutil.rmtree(save_to_dir)

    os.makedirs(save_to_dir)

    print('Saving parameters %s to %s' % (params, params_filepath))
    with open(params_filepath, "wt") as f:
        json.dump(json.loads(jsonpickle.encode(params)), f, indent=4)


def get_params(name, epoch):
    save_to_dir = get_train_dir(name, epoch)
    params_filepath = os.path.join(save_to_dir, 'params.json')
    with open(params_filepath, "r") as f:
        result = jsonpickle.decode(f.read())
        return result['name'], result['epoch'], result['time']


