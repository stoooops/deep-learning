#!/usr/bin/env python

def prefix_print_fn(print_fn, prefix):
    return lambda *argv, **kwargs: print_fn(('%s ' + argv[0]) % prefix, *argv[1:], **kwargs)
