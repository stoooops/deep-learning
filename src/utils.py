#!/usr/bin/env python

import os


def print_list(l):
    print('\n'.join([str(item) for item in l]))

def properties_names(obj):
    return sorted([k for k, v in obj.__dict__.items()])


SRC_DIR = os.path.dirname(os.path.realpath(__file__))
assert SRC_DIR[-3:] == 'src'

MODELS_DIR = os.path.join(SRC_DIR, '../models')

