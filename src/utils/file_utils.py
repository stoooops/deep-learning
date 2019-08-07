#!/usr/bin/env python

import os


SRC_DIR = os.path.dirname(os.path.realpath(__file__))
_EXPECTED_END_SRC_DIR = 'src/utils'
assert SRC_DIR[-len(_EXPECTED_END_SRC_DIR):] == _EXPECTED_END_SRC_DIR

MODELS_DIR = os.path.normpath(os.path.join(SRC_DIR, '../../models'))

