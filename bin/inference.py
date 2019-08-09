#!/usr/bin/env python

import sys
import time
import argparse

from src.meta.tensor_apis import TensorApi
from src.utils.logger import HuliLogging
from bin.train import get_data, get_model, MODEL_NAMES

import tensorflow as tf

logger = HuliLogging.get_logger(__name__)
HuliLogging.attach_stdout()


logger.info('=' * 50)
logger.info('tensorflow-%s' % tf.__version__)
logger.info('=' * 50)

DEFAULT_WARM_UP = 25
DEFAULT_TRIALS = 100
DEFAULT_REPEAT = 3

DEFAULT_KERAS_INFERENCE = True
DEFAULT_TFLITE_INFERENCE = True
DEFAULT_TRT_INFERENCE = True


def get_time_format(num):
    return '%3d' if num >= 100 else '%2d' if num >= 10 else '%d'


def predict(model, test_images, trials, log_first=0, log_last=0):
    start = time.time()
    longest_line = 0
    for i in range(0, trials):
        now = time.time()
        model.predict(test_images)
        elapsed = time.time() - now
        running_avg = (time.time() - start) / (i + 1)

        time_format = get_time_format(trials)
        log = (time_format + ': %.3fs') % (i + 1, elapsed)
        if i < log_first or i > trials - log_last:
            print(log)
        else:
            if i + 1 != trials:
                log += ' ['
                log += '=' * i
                log += '>'
                log += '.' * (trials - i - 1)
                log += ']'
                remaining_time = running_avg * (trials - i - 1)
                log += ' ETA: %.3fs' % remaining_time
            longest_line = max(len(log), longest_line)
            if len(log) < longest_line:
                log += ' ' * (longest_line - len(log))
            print('\r%s' % log, end='\r')

        yield elapsed


def warm_up(model, test_images, trials, time_format):
    logger.info('%s Warming up on %d images for %d iterations...' % (model.name, len(test_images), trials))
    timings = [p for p in predict(model, test_images, trials)]
    elapsed = sum(timings)
    log_images_size = ('%d' % len(test_images)).ljust(5)
    log_prefix = ('[' + time_format + 'x%s %s]') % (len(timings), log_images_size, model.mode)
    logger.info('%s Warm up =   %.3fs (avg = %.3fs)' % (log_prefix, elapsed, elapsed / len(timings)))


def test(model, test_images, trials, repeat, time_format):
    if len(test_images) == 1:
        phrase = 'single image'
    else:
        phrase = 'batch of %d images' % len(test_images)
    logger.info('%s Repeat %dx timing on %s for %d iterations...' % (model.name, repeat, phrase, trials))
    for i in range(repeat):
        timings = [p for p in predict(model, test_images, trials, log_first=0)]
        elapsed = sum(timings)
        log_images_size = ('%d' % len(test_images)).ljust(5)
        log_prefix = ('[' + time_format + 'x%s %s]') % (len(timings), log_images_size, model.mode)
        logger.info('%s Test time = %.3fs (avg = %.3fs)' % (log_prefix, elapsed, elapsed / len(timings)))


def infer(model, test_images, keras=DEFAULT_KERAS_INFERENCE, tflite=DEFAULT_TFLITE_INFERENCE,
          trt=DEFAULT_TRT_INFERENCE, warm_up_trials=DEFAULT_WARM_UP, trials=DEFAULT_TRIALS, repeat=DEFAULT_REPEAT):
    time_format = get_time_format(len(test_images))

    # Test multiple sizes
    for i in [1, 10, 100, 1000]:
        def run(mode):
            ret = model.change_mode(mode)
            if ret != 0:
                logger.error('Changing to mode %s failed with error %d', TensorApi.KERAS, ret)
                exit(1)
            warm_up(model, test_images[:i], warm_up_trials, time_format)
            test(model, test_images[:i], trials, repeat, time_format)

        if keras:
            run(TensorApi.KERAS)
        if tflite:
            run(TensorApi.TF_LITE)
        if trt:
            run(TensorApi.TENSOR_RT)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('-e', '--epoch', type=int, default=1, help='Training epoch.')
    p.add_argument('-m', '--model', required=True, help='Model name. One of %s' % MODEL_NAMES)
    p.add_argument('-w', '--warm-up', type=int, default=DEFAULT_WARM_UP, help='Warmup trials')
    p.add_argument('-t', '--trials', type=int, default=DEFAULT_TRIALS, help='Test trials')
    p.add_argument('-r', '--repeat', type=int, default=DEFAULT_REPEAT, help='Repeat sceneratios')
    p.add_argument('--skip-keras', default=not DEFAULT_KERAS_INFERENCE, help='Skip keras inference',
                   action="store_true")
    p.add_argument('--skip-tflite', default=not DEFAULT_TFLITE_INFERENCE, help='Skip tflite inference',
                   action="store_true")
    p.add_argument('--skip-trt', default=not DEFAULT_TRT_INFERENCE, help='Skip trt inference', action="store_true")
    args = p.parse_args()
    assert args.model in MODEL_NAMES
    assert args.epoch > 0
    assert args.warm_up >= 0
    assert args.trials >= 1
    assert args.repeat >= 1
    return args


def main():
    # Args
    args = get_args()
    model_name = args.model
    epoch = args.epoch
    warm_up_trials = args.warm_up
    trials = args.trials
    repeat = args.repeat

    # Load Data
    (train_images, train_labels), (test_images, test_labels) = get_data()

    # Load Model
    ret, model = get_model(model_name, epoch)
    if ret != 0:
        logger.error('Getting model failed with error %d', ret)
        exit(1)
    model.summary()

    infer(model, test_images, keras=not args.skip_keras, tflite=not args.skip_tflite, trt=not args.skip_trt,
          warm_up_trials=warm_up_trials, trials=trials, repeat=repeat)


if __name__ == '__main__':
    now = time.time()
    logger.info('')
    logger.info('')
    logger.info('> ' + ' '.join(sys.argv))
    try:
        main()
    except Exception as e:
        logger.exception(e)
        exit(1)

    logger.info('')
    logger.info('> ' + ' '.join(sys.argv))
    logger.info('[%.3fs] SUCCESS!!!', time.time() - now)
