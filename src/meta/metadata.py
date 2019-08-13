#!/usr/bin/env python

import os
import json
import jsonpickle
from src.meta.constants import UNKNOWN_EPOCH
from src.meta.errors import ERROR_TF_META_FILE_NOT_FOUND
from src.utils import file_utils

from src.utils.logger import Logging

logger = Logging.get_logger(__name__)


class Metadata:
    """
    Wrapper class for metadata about a model

    Mutable.
    """
    def __init__(self, name, epoch=UNKNOWN_EPOCH, input_names=None, output_names=None):
        assert name is not None and isinstance(name, str)
        self.name = name
        assert epoch is not None and isinstance(epoch, int) and (epoch == UNKNOWN_EPOCH or epoch >= 0)
        self.epoch = epoch
        self.input_names = input_names
        self.output_names = output_names

    @staticmethod
    def from_md(filepath_md):
        """Load the metadata file and json decode it."""
        assert filepath_md is not None
        assert os.path.splitext(filepath_md)[1] == file_utils.EXTENSION_MD
        assert os.path.exists(filepath_md)

        try:
            with open(filepath_md, "r") as f:
                result = jsonpickle.decode(f.read())
        except Exception as e:
            logger.exception('Failed loading metadata filed from %s: %s', filepath_md, e)
            return ERROR_TF_META_FILE_NOT_FOUND, None

        return 0, result

    def save(self, filepath):
        assert os.path.splitext(filepath)[1] == file_utils.EXTENSION_MD

        if os.path.exists(filepath):
            logger.warn('Overwriting %s...', filepath)

        with open(filepath, "wt") as f:
            json.dump(json.loads(jsonpickle.encode(self)), f, indent=4)

        return 0

    def update_epoch(self, epoch, prefix=None):
        if self.epoch != epoch:
            logger.debug('%s Updating epoch to %d', prefix or self.name, epoch)
            self.epoch = epoch

    def dump(self, prefix=None):
        logger.debug('%s epoch = %s', prefix or self.name, self.epoch)
        logger.debug('%s inputs = %s', prefix or self.name, self.input_names)
        logger.debug('%s outputs = %s', prefix or self.name, self.output_names)

    # Directory

    def file_dir(self, epoch=None):
        epoch = epoch if epoch is not None else self.epoch
        result = file_utils.model_dir(self.name, epoch)
        if not os.path.exists(result):
            os.makedirs(result)
        return result

    # Filename

    def filename_no_ext(self, epoch=None):
        epoch = epoch if epoch is not None else self.epoch
        return file_utils.model_filename_no_ext(self.name, epoch)

    # .h5 - architecture/weights/optimizer

    def filename_h5(self, epoch=None):
        epoch = epoch if epoch is not None else self.epoch
        return file_utils.model_filename_h5(self.name, epoch)

    def filepath_h5(self, epoch=None, dir_=None):
        epoch = epoch if epoch is not None else self.epoch
        return file_utils.model_filepath_h5(self.name, epoch, dir_=dir_)

    # .h5 - architecture/weights

    def filename_no_opt_h5(self, epoch=None):
        epoch = epoch if epoch is not None else self.epoch
        return file_utils.model_filename_no_opt_h5(self.name, epoch)

    def filepath_no_opt_h5(self, epoch=None, dir_=None):
        epoch = epoch if epoch is not None else self.epoch
        return file_utils.model_filepath_no_opt_h5(self.name, epoch, dir_=dir_)

    # .h5 - weights

    def filename_weights_h5(self, epoch=None):
        epoch = epoch if epoch is not None else self.epoch
        return file_utils.model_filename_weights_h5(self.name, epoch)

    def filepath_weights_h5(self, epoch=None, dir_=None):
        epoch = epoch if epoch is not None else self.epoch
        return file_utils.model_filepath_weights_h5(self.name, epoch, dir_=dir_)

    # .md - metadata

    def filename_md(self, epoch=None):
        epoch = epoch if epoch is not None else self.epoch
        return file_utils.model_filename_md(self.name, epoch)

    def filepath_md(self, epoch=None, dir_=None):
        epoch = epoch if epoch is not None else self.epoch
        return file_utils.model_filepath_md(self.name, epoch, dir_=dir_)

    # .pb

    def filename_pb(self, epoch=None):
        epoch = epoch if epoch is not None else self.epoch
        return file_utils.model_filename_pb(self.name, epoch)

    def filepath_pb(self, epoch=None, dir_=None):
        epoch = epoch if epoch is not None else self.epoch
        return file_utils.model_filepath_pb(self.name, epoch, dir_=dir_)

    # .tflite - INT8

    def filename_tflite(self, epoch=None):
        epoch = epoch if epoch is not None else self.epoch
        return file_utils.model_filename_tflite(self.name, epoch)

    def filepath_tflite(self, epoch=None, dir_=None):
        epoch = epoch if epoch is not None else self.epoch
        return file_utils.model_filepath_tflite(self.name, epoch, dir_=dir_)
