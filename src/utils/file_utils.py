#!/usr/bin/env python

import os


FILE_DIR = os.path.dirname(os.path.realpath(__file__))
_EXPECTED_END_FILE_DIR = 'src/utils'
assert FILE_DIR[-len(_EXPECTED_END_FILE_DIR):] == _EXPECTED_END_FILE_DIR

SRC_DIR = os.path.normpath(os.path.join(FILE_DIR, '..'))

MODELS_DIR = os.path.normpath(os.path.join(SRC_DIR, '../models'))

TMP_DIR = os.path.normpath(os.path.join(SRC_DIR, '../tmp'))

LOG_DIR = os.path.normpath(os.path.join(TMP_DIR, 'logs'))
