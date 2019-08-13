#!/usr/bin/env python


import os
import tensorflow as tf

from .errors import ERROR_TF_CAUGHT_EXCEPTION
from src.meta.inference import InferenceModel
from src.meta.metadata import Metadata
from src.utils.logger import Logging
from src.utils import file_utils

logger = Logging.get_logger(__name__)


class TensorFlowModel(InferenceModel):

    def __init__(self, metadata, graph_def, graph):
        """
        :type graph_def: tensorflow.GraphDef
        :type epoch: int
        """
        assert metadata is not None and isinstance(metadata, Metadata)
        self.name = metadata.name
        self.metadata = metadata

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

        self.session = tf.Session()

    def __del__(self):
        logger.info('%s Closing session...', self.name)
        self.session.close()

    def predict(self, *argv, **kwargs):
        assert len(argv) == 1
        x = argv[0]
        ret, y = 0, None
        try:
            with self.graph.as_default():
                # Dimensions to the model are batch, height, width, colors. sample image needs to have batch
                # axes prepended to the shape so we expand dims.
                y = self.session.run(self.graph_y, feed_dict={self.graph_x: x})
        except RuntimeError as e:
            logger.exception('%s session is in invalid state (e.g. has been closed).', e)
            ret = ERROR_TF_CAUGHT_EXCEPTION
        except TypeError as e:
            logger.exception('Given `fetches` or `feed_dict` keys are of an inappropriate type.', e)
            ret = ERROR_TF_CAUGHT_EXCEPTION
        except ValueError as e:
            logger.exception(
                'Given `fetches` or `feed_dict` keys are invalid or refer to a `Tensor` that doesn\'t exist.', e)
            ret = ERROR_TF_CAUGHT_EXCEPTION
        if ret != 0:
            return ret, None

        return ret, y

    def dump(self):
        graph_node_names = [n.name for n in self.graph_def.node]
        logger.debug('%s graph_def.node names: %s', self.name, graph_node_names)

        operations = self.graph.get_operations()
        logger.debug('%s graph.get_operations(): [%d] %s', self.name, len(operations), [o.name for o in operations])

        return 0

    @staticmethod
    def from_pb(filepath_md, filepath_pb):
        assert filepath_md is not None and isinstance(filepath_md, str)
        assert os.path.splitext(filepath_md)[1] == file_utils.EXTENSION_MD
        assert filepath_pb is not None and isinstance(filepath_pb, str)
        assert os.path.splitext(filepath_pb)[1] == file_utils.EXTENSION_PB

        logger.debug('Loading metadata from %s...', filepath_md)
        ret, metadata = Metadata.from_md(filepath_md)
        if ret != 0:
            return ret

        logger.debug('%s Loading frozen graph model from %s...', metadata.name, filepath_pb)
        tf.reset_default_graph()

        # load graph
        with tf.Session() as sess:
            with tf.gfile.GFile(filepath_pb, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                # don't use "with tf.Graph().as_default() as graph" here or tensorboard will fail for some reason
                tf.import_graph_def(graph_def, name=metadata.name)

        # write tensorboard info
        tensorboard_log_dir = file_utils.tensorboard_log_dir(metadata.name)
        train_writer = tf.summary.FileWriter(tensorboard_log_dir)
        train_writer.add_graph(sess.graph)
        train_writer.flush()
        train_writer.close()
        logger.debug('%s pb model imported into tensorboard. Visualize by running: ', metadata.name)
        logger.debug('> tensorboard --logdir=%s', tensorboard_log_dir)

        result = TensorFlowModel(metadata, graph_def, sess.graph)

        return 0, result
