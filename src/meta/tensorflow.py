#!/usr/bin/env python


import tensorflow as tf

from src.meta.constants import UNKNOWN_EPOCH
from src.meta.errors import *
from src.meta.tensor_apis import AbstractTensorModel, TensorApi
from src.utils.logger import HuliLogging

logger = HuliLogging.get_logger(__name__)


class TensorFlowModel(AbstractTensorModel):

    def __init__(self, name, graph_def, epoch=UNKNOWN_EPOCH):
        """
        :type name: str
        :type graph_def: tensorflow.GraphDef
        :type epoch: int
        """
        super().__init__(name)

        assert isinstance(epoch, int) and (epoch == UNKNOWN_EPOCH or epoch >= 1)
        self.epoch = epoch

        # GraphDef Model
        assert graph_def is not None and isinstance(graph_def, tf.GraphDef),\
            'Expected tf.GraphDef but got: %s' % graph_def
        self.graph_def = graph_def

        self.mode = TensorApi.TENSOR_FLOW

    def compile(self):
        return ERROR_TF_META_UNIMPLEMENTED

    def fit(self):
        return ERROR_TF_META_UNIMPLEMENTED

    def evaluate(self):
        return ERROR_TF_META_UNIMPLEMENTED

    def predict(self):
        return ERROR_TF_META_UNIMPLEMENTED

    def save(self):
        return ERROR_TF_META_UNIMPLEMENTED

    def dump(self):
        logger.debug('%s Nothing to dump!!!!', self.name)

    @staticmethod
    def load(name, epoch, filepath):
        logger.debug('%s Loading frozen graph model from %s...', name, filepath)

        with tf.Session() as sess:
            # First deserialize your frozen graph:
            logger.info('%s Loading frozen graph from %s...', name, filepath)
            with tf.gfile.GFile(filepath, 'rb') as f:
                frozen_graph = tf.GraphDef()
                frozen_graph.ParseFromString(f.read())

            # Then, we import the graph_def into a new Graph and return it
            with tf.Graph().as_default() as graph:
                # The name var will prefix every op/nodes in your graph
                # Since we load everything in a new graph, this is not needed
                tf.import_graph_def(frozen_graph, name=name)

        result = TensorFlowModel(name, graph, epoch=epoch)

        return 0, result

