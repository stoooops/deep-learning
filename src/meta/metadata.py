#!/usr/bin/env python


from src.meta.constants import UNKNOWN_EPOCH

from src.utils.logger import HuliLogging

logger = HuliLogging.get_logger(__name__)


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

    def dump(self, prefix=None):
        logger.debug('%s epoch = %s', prefix or self.name, self.epoch)
        logger.debug('%s inputs = %s', prefix or self.name, self.input_names)
        logger.debug('%s outputs = %s', prefix or self.name, self.output_names)
