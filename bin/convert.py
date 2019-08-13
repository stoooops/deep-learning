#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import os
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf

# Helper libraries
import os
import sys
import time
import argparse

from src.data import cifar100
from src.convert.keras import freeze_graph, convert_tf_lite
from src.utils.color_utils import bcolors
from src.utils.logger import Logging
from src.utils import file_utils
Logging.attach_stdout()
# HuliLogging.debug_dim()
# HuliLogging.info_blue()
# HuliLogging.warn_yellow()
# HuliLogging.error_red()

SUPPORTED_INPUTS = [file_utils.EXTENSION_H5]

SUPPORTED_OUTPUTS = [file_utils.EXTENSION_PB, os.path.splitext(file_utils.EXTENSION_INT8_TFLITE)[1]]

logger = Logging.get_logger(__name__)


def line():
    return '=' * 50


print(line())
print(tf.__name__, '-', tf.__version__, sep='')
print(line())


def log_bold(*argv, **kwargs):
    logger.info(line() + ' ' + argv[0] + ' ' + line(), *argv[1:], **kwargs)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input', help="Model to use", required=True)
    p.add_argument('-o', '--output', help="Location to write output model to", required=True)
    p.add_argument('-d', '--representative_data',
                   help="Name of representative data (required for quantization conversions)", required=False)
    args = p.parse_args()
    assert os.path.splitext(args.input)[1] in SUPPORTED_INPUTS, '%s not in %s' % (args.input, SUPPORTED_INPUTS)
    assert os.path.splitext(args.output)[1] in SUPPORTED_OUTPUTS, '%s not in %s' % (args.output, SUPPORTED_OUTPUTS)
    assert args.output[-len(file_utils.EXTENSION_INT8_TFLITE):] != file_utils.EXTENSION_INT8_TFLITE \
        or args.representative_data is not None, \
        'Must pass representative data for %s conversion' % file_utils.EXTENSION_INT8_TFLITE
    return args


def get_data(representative_data):
    if representative_data is None:
        return None
    elif representative_data == cifar100.NAME:
        (train_x, train_y), (test_x, test_y) = cifar100.load_data()
        return train_x
    else:
        assert 1 == 0, 'Unexpected representative_data: %s' % representative_data


def convert(input, output, representative_data=None):
    input_ext = os.path.splitext(input)[1]
    assert input_ext == file_utils.EXTENSION_H5, 'Unexpected input file type: %s' % input

    output_ext = os.path.splitext(output)[1]
    assert output_ext in [file_utils.EXTENSION_PB, os.path.splitext(file_utils.EXTENSION_INT8_TFLITE)[1]],\
        'Unexpected output file type: %s' % output

    if output_ext == file_utils.EXTENSION_PB:
        if representative_data is not None:
            logger.warn('No need to pass representative data for %s conversion', file_utils.EXTENSION_PB)
        ret = freeze_graph(input, output)
    elif output_ext == os.path.splitext(file_utils.EXTENSION_INT8_TFLITE)[1]:
        assert representative_data is not None, 'representative_data is None'
        ret = convert_tf_lite(input, output, representative_data=representative_data)
    else:
        ret = -1
    return ret


def main():
    # Args
    log_bold('PARSE')
    args = get_args()
    input_ = args.input
    output = args.output
    representative_data = get_data(args.representative_data)

    ret = convert(input_, output, representative_data=representative_data)
    if ret != 0:
        logger.error('Convert failed with error %d', ret)
        return ret

    return 0


if __name__ == '__main__':
    now_ = time.time()
    logger.info('')
    logger.info('')
    bcolors.light_cyan(logger.info, '> ' + ' '.join(sys.argv))

    ret_ = 0
    try:
        main()
    except Exception as e:
        logger.exception('Uncaught exception: %s', e)
        ret_ = 1
    logger.info('')
    bcolors.light_cyan(logger.info, '> ' + ' '.join(sys.argv))
    logger.info('')

    if ret_ == 0:
        bcolors.light_green(logger.info, '[%.3fs] SUCCESS!!!',time.time() - now_)
    else:
        bcolors.light_red(logger.error, '[%.3fs] FAIL!!!', time.time() - now_)

    exit(ret_)
