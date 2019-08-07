#!/usr/bin/env python


def print_list(l):
    print('\n'.join([str(item) for item in l]))


def properties_names(obj):
    return sorted([k for k, v in obj.__dict__.items()])

