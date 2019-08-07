#!/usr/bin/env python

import sys
import time
import argparse
import numpy as np

from src.logger import HuliLogging
from src.train import get_data, get_filepath, get_model, MODEL_NAMES, MODEL_RESNET50

logger = HuliLogging.get_logger(__name__)

import tensorflow as tf
logger.info('=' * 30)
logger.info('tensorflow-%s' % tf.__version__)
logger.info('=' * 30)

DEFAULT_WARM_UP = 25
DEFAULT_TRIALS = 100
DEFAULT_REPEAT = 3
OLD = 'old'
NEW = 'new'

BATCH_SIZE = 10


def do_evaluate(model, test_images):
    return model.predict(test_images)


def get_time_format(num):
    return '%3d' if num >= 100 else '%2d' if num >= 10 else '%d'


def predict(model, test_images, f_evaluate, trials, log_first=0, log_last=0):
    start = time.time()
    longest_line = 0
    for i in range(0, trials):
        now = time.time()
        f_evaluate(model, test_images)
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


def warm_up(model_name, model, test_images, f_evaluate, style, trials, time_format):
    logger.info('Warming up MODEL[%s] on %d images for %d iterations...' % (model_name, len(test_images), trials))
    timings = [p for p in predict(model, test_images, f_evaluate, trials)]
    elapsed = sum(timings)
    log_images_size = ('%d' % len(test_images)).ljust(5)
    log_prefix = ('[' + time_format + 'x%s %s]') % (len(timings), log_images_size, style)
    logger.info('%s Warm up =   %.3fs (avg = %.3fs)' % (log_prefix, elapsed, elapsed / len(timings)))


def test(model_name, model, test_images, f_evaluate, style, trials, repeat, time_format):
    if len(test_images) == 1:
        phrase = 'single image'
    else:
        phrase = 'batch of %d images' % len(test_images)
    logger.info('Repeat %dx timing MODEL[%s] on %s for %d iterations...' % (repeat, model_name, phrase, trials))
    for i in range(repeat):
        timings = [p for p in predict(model, test_images, f_evaluate, trials, log_first=0)]
        elapsed = sum(timings)
        log_images_size = ('%d' % len(test_images)).ljust(5)
        log_prefix = ('[' + time_format + 'x%s %s]') % (len(timings), log_images_size, style)
        logger.info('%s Test time = %.3fs (avg = %.3fs)' % (log_prefix, elapsed, elapsed / len(timings)))


def infer(model_name, model, test_images, f_evaluate, style,
          warm_up_trials=DEFAULT_WARM_UP, trials=DEFAULT_TRIALS, repeat=DEFAULT_REPEAT):
    time_format = get_time_format(max(trials, warm_up_trials))

    # Test multiple sizes
    for i in [1, 10, 100, 1000, 10000]:
        warm_up(model_name, model, test_images[:i], f_evaluate, style, warm_up_trials, time_format)
        test(model_name, model, test_images[:i], f_evaluate, style, trials, repeat, time_format)


def to_tflite(filepath):
    return filepath.replace('.h5', '.tflite')


def convert_model(model_name, epoch, train_images):
    def representative_dataset_gen():
        for i in range(1000):
            yield [train_images[i: i + 1].astype(np.float32)]

    filepath = get_filepath(model_name, epoch)

    # Currentry, it seems that tf.lite.TFLiteConverter does not suppert inference_input/output_type yet.
    # So we have to use tf.compat.v1.lite.TFLiteConverter.
    logger.info('Creating TFLiteConverter from keras model file %s...', filepath)
    #input_shape = {'resnet50': (None, 32, 32, 3)}
    #logger.info('Using input shape %s', input_shape)
    converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(filepath)
    converter.representative_dataset = representative_dataset_gen
    target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8] # throw error if conversion not available
    if tf.__version__[0] == '2':
        converter.target_spec.supported_ops = target_ops
    else:
        converter.target_ops = target_ops
    logger.info('Converting with supported ops %s', target_ops)
    converter.inference_input_type = tf.uint8
    logger.info('Converting to inference input type %s', converter.inference_input_type)
    converter.inference_output_type = tf.uint8
    logger.info('Converting to inference output type %s', converter.inference_output_type)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # seems that result file has same size no matter what
    logger.info('Converting using %s optimizations', converter.optimizations)

    logger.info('Converting to tflite model...')
    tflite_model = converter.convert()
    fileout = to_tflite(filepath)
    logger.info('Writing tflite model to %s...', fileout)
    with open(fileout, 'wb') as o_:
        o_.write(tflite_model)

    interpreter = tf.lite.Interpreter(model_path=fileout)
    interpreter.allocate_tensors()

    #import ipdb; ipdb.set_trace()

    return interpreter


def get_evaluate_func(do_new_api, model_name, epoch, train_images):
    if not do_new_api:
        return do_evaluate, OLD
    else:
        interpreter = convert_model(model_name, epoch, train_images)

        input_detail = interpreter.get_input_details()[0]
        in_std, in_mean = input_detail['quantization']
        in_index = input_detail['index']
        logger.info('in_std, in_mean = (%.2f, %.2f)' % (in_std, in_mean))
        logger.info('in_shape = %s', input_detail['shape'])
        logger.info('in_index = %s', input_detail['index'])
        output_detail = interpreter.get_output_details()[0]
        out_std, out_mean = output_detail['quantization']
        out_index = output_detail['index']

        def quantize_in(real_value):
            return (real_value / in_std + in_mean).astype(np.uint8)

        def quantize_out(real_value):
            return (real_value.astype(np.float32) - out_mean) * in_std

        def f_evaluate(model, test_images):
            quantize_test_images = quantize_in(test_images)

            result = []
            for i in range(0, len(quantize_test_images), 1):

                interpreter.set_tensor(in_index, quantize_test_images[i:i+1])
                interpreter.invoke()
                quantized_result = interpreter.get_tensor(out_index)

                result.append(quantize_out(quantized_result))

            return result

        return f_evaluate, NEW


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('-x', '--do-test', action='store_true')
    p.add_argument('-e', '--epoch', type=int, default=1, help='Training epoch.')
    p.add_argument('-m', '--model', required=True, help='Model name. One of %s' % MODEL_NAMES)
    p.add_argument('-w', '--warm-up', type=int, default=DEFAULT_WARM_UP, help='Warmup trials')
    p.add_argument('-t', '--trials', type=int, default=DEFAULT_TRIALS, help='Test trials')
    p.add_argument('-r', '--repeat', type=int, default=DEFAULT_REPEAT, help='Repeat sceneratios')
    p.add_argument('-i', '--inference', action="store_true")
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
    model = get_model(args.model, epoch)
    model.summary()
    logger.info('Input tensor: %s', model.inputs[0])

    if args.do_test:
        # Get the concrete function from the Keras model.
        run_model = tf.function(lambda x: model(x))

        # Save the concrete function.
        concrete_func = run_model.get_concrete_function(
            tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
        )

        # Convert the model to standard TensorFlow Lite model
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converted_tflite_model = converter.convert()
        open('model.tflite', "wb").write(converted_tflite_model)

        # Convert the model to quantized version with post-training quantization
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        tflite_quant_model = converter.convert()
        open('model_quant.tflite', "wb").write(tflite_quant_model)

    # print('NORMAL')
    # result = do_evaluate(model, test_images[:1])
    # print(result)
    # print(result.sum())
    # print(result.argmax())
    # print('DONE')
    f_evaluate, style = get_evaluate_func(args.inference, model_name, epoch, train_images)

    # if args.inference:
    #     print('NEW')
    #     result = f_evaluate(model, test_images[:1])
    #     print(result)
    #     print(result.sum())
    #     print(result.argmax())
    #     print('DONE')

    infer(model_name, model, test_images, f_evaluate, style,
          warm_up_trials=warm_up_trials, trials=trials, repeat=repeat)


if __name__ == '__main__':
    logger.info('')
    logger.info('')
    logger.info('> ' + ' '.join(sys.argv))
    main()
