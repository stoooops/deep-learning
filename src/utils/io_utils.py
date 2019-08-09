#!/usr/bin/env python

def prefix_print_fn(print_fn, prefix):
    return lambda *argv, **kwargs: print_fn(prefix + ' ' + argv[0], *argv[1:], **kwargs)
def postfix_print_fn(print_fn, postfix):
    return lambda *argv, **kwargs: print_fn(argv[0] + ' ' + postfix, *argv[1:], **kwargs)

def pre_postfix_print_fn(print_fn, prefix, postfix):
    return lambda *argv, **kwargs: print_fn(prefix + ' ' + argv[0] + ' ' + postfix, *argv[1:], **kwargs)
