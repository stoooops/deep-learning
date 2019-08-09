#!/usr/bin/env python


import os
from datetime import datetime
import tensorflow as tf

from src.meta.constants import TENSORBOARD_DIR
from src.meta.errors import *
from src.meta.metadata import Metadata
from src.meta.tensor_apis import AbstractTensorModel, TensorApi
from src.utils.logger import HuliLogging

logger = HuliLogging.get_logger(__name__)


class TensorFlowModel(AbstractTensorModel):

    def __init__(self, name, metadata, graph_def, graph):
        """
        :type name: str
        :type graph_def: tensorflow.GraphDef
        :type epoch: int
        """
        super().__init__(name, metadata, mode=TensorApi.TENSORFLOW)

        # GraphDef
        assert graph_def is not None and isinstance(graph_def, tf.GraphDef),\
            'Expected tf.GraphDef but got: %s' % graph_def
        self.graph_def = graph_def

        # Graph
        assert graph is not None and isinstance(graph, tf.Graph),\
            'Expected tf.Graph but got: %s' % graph
        self.graph = graph

        # input x
        assert self.metadata.input_names is not None and len(self.metadata.input_names) > 0
        self.input_names = [self.name + '/' + input_name for input_name in self.metadata.input_names]
        self.graph_x = [self.graph.get_tensor_by_name(name) for name in self.input_names]
        assert len(self.graph_x) == 1, 'MULTIPLE INPUTS NOT SUPPORTED YET.'
        self.graph_x = self.graph_x[0]
        # output y
        assert self.metadata.output_names is not None and len(self.metadata.output_names) > 0
        self.output_names = [self.name + '/' + output_name for output_name in self.metadata.output_names]
        self.graph_y = [self.graph.get_tensor_by_name(name) for name in self.output_names]

    def compile(self):
        return ERROR_TF_META_UNIMPLEMENTED

    def fit(self):
        return ERROR_TF_META_UNIMPLEMENTED

    def evaluate(self):
        return ERROR_TF_META_UNIMPLEMENTED

    def predict(self, *argv, **kwargs):
        assert len(argv) == 1
        x = argv[0]
        ret, y = 0, None
        try:
            with self.graph.as_default():
                # Dimensions to the model are batch, height, width, colors. sample image needs to have batch
                # axes prepended to the shape so we expand dims.
                y = tf.Session().run(self.graph_y, feed_dict={self.graph_x: x})
        except RuntimeError as e:
            logger.exception('%s session is in invalid state (e.g. has been closed).', e)
            ret = ERROR_TF_META_CAUGHT_EXCEPTION
        except TypeError as e:
            logger.exception('Given `fetches` or `feed_dict` keys are of an inappropriate type.', e)
            ret = ERROR_TF_META_CAUGHT_EXCEPTION
        except ValueError as e:
            logger.exception(
                'Given `fetches` or `feed_dict` keys are invalid or refer to a `Tensor` that doesn\'t exist.', e)
            ret = ERROR_TF_META_CAUGHT_EXCEPTION
        if ret != 0:
            return ret, None

        return ret, y

    def save(self):
        return ERROR_TF_META_UNIMPLEMENTED

    def dump(self):
        graph_node_names = [n.name for n in self.graph_def.node]
        logger.debug('%s graph_def.node names: %s', self.log_prefix(), graph_node_names)

        operations = self.graph.get_operations()
        logger.debug('%s graph.get_operations(): [%d] %s', self.log_prefix(), len(operations), [o.name for o in operations])

        return 0

    @staticmethod
    def load(name, metadata, filepath):
        assert name is not None and isinstance(name, str)
        assert metadata is not None and isinstance(metadata, Metadata)
        assert filepath is not None and isinstance(filepath, str)

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

        result = TensorFlowModel(name, metadata, graph_def, sess.graph)

        return 0, result
