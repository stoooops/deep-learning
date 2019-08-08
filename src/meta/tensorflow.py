#!/usr/bin/env python


import os
from datetime import datetime
import tensorflow as tf

from src.meta.constants import TENSORBOARD_DIR, UNKNOWN_EPOCH
from src.meta.errors import *
from src.meta.tensor_apis import AbstractTensorModel, TensorApi
from src.utils.logger import HuliLogging

logger = HuliLogging.get_logger(__name__)


class TensorFlowModel(AbstractTensorModel):

    def __init__(self, name, graph_def, graph, epoch=UNKNOWN_EPOCH):
        """
        :type name: str
        :type graph_def: tensorflow.GraphDef
        :type epoch: int
        """
        super().__init__(name)

        assert isinstance(epoch, int) and (epoch == UNKNOWN_EPOCH or epoch >= 1)
        self.epoch = epoch

        # GraphDef
        assert graph_def is not None and isinstance(graph_def, tf.GraphDef),\
            'Expected tf.GraphDef but got: %s' % graph_def
        self.graph_def = graph_def

        # Graph
        assert graph is not None and isinstance(graph, tf.Graph),\
            'Expected tf.Graph but got: %s' % graph
        self.graph = graph

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
        graph_node_names = [n.name for n in self.graph_def.node]
        logger.debug('%s graph_def.node names: %s', self.name, graph_node_names)

        operations = self.graph.get_operations()
        logger.debug('%s graph.get_operations(): [%d] %s', self.name, len(operations), [o.name for o in operations])

        return 0


    @staticmethod
    def load(name, epoch, filepath):
        logger.debug('%s Loading frozen graph model from %s...', name, filepath)

        tf.reset_default_graph()

        # load graph
        with tf.Session() as sess:
            with tf.gfile.GFile(filepath, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                # don't use "with tf.Graph().as_default() as graph" here or tensorboard will fail for some reason
                tf.import_graph_def(graph_def, name=name)

        # write tensorboard info
        tensorboard_log_dir = os.path.join(TENSORBOARD_DIR,
                                           datetime.now().strftime("%Y%m%d-%H%M%S") + '_' + name)
        train_writer = tf.summary.FileWriter(tensorboard_log_dir)
        train_writer.add_graph(sess.graph)
        train_writer.flush()
        train_writer.close()
        logger.debug('%s pb model imported into tensorboard. Visualize by running: ', name)
        logger.debug('> tensorboard --logdir=%s', tensorboard_log_dir)

        result = TensorFlowModel(name, graph_def, sess.graph, epoch=epoch)

        return 0, result
