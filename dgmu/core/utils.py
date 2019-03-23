"""dGMU core utilities module"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import errno
import numpy as np
import tensorflow as tf



def get_batch(X1, X2, Y_, size):
    """ Batch generator

    :param X1: training data
    :param X2: test/validation data
    :param Y_: training labels
    :param size: batch size
    :return: respective data in batch of defined size
    """

    a = np.random.choice(len(X1), size, replace=False)

    return X1[a], X2[a], Y_[a]

class Score(object):
    """Classification scoring methods"""

    @staticmethod
    def accuracy(y, y_, summary=True, name="accuracy"):
        #Sampled from yadlt/core/layers.py
        """Computes accuracy
        :param y: tf.Tensor Current label prediction
        :param y_: tf.Tensor True labels
        :param summary: bool, if True, saves tf summary for the operation
        :return: returns the accuracy operation tensor
        """

        with tf.name_scope(name):
            mod_pred = tf.argmax(mod_y, 1)
            correct_pred = tf.equal(mod_pred, tf.argmax(ref_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            if summary:
                tf.summary.scalar('accuracy', accuracy)
            return accuracy

class Borg:
    """Simulate the behaviour of a singleton"""
    _shared_state = {}
    def __init__(self):
        self.__dict__ = self._shared_state

class Config(Borg):
    """Model directory singleton class

    :param models_dir: string, name of the folder for stored models, relative to self.root_dir
    :param data_dir: string, name of the data folder, relative to self.root_dir
    :param logs_dir: string, name of the logs folder, relative to self.root_dir
    """
    def __init__(self, models_dir='stored_models/', data_dir='sequence_data/', logs_dir='logs/'):
        Borg.__init__(self)

        self.root_dir = os.getcwd()
        self.models_dir = os.path.join(self.root_dir,models_dir)
        self.data_dir = os.path.join(self.root_dir,data_dir)
        self.logs_dir = os.path.join(self.root_dir,logs_dir)
        self._mkdir(self.models_dir)
        self._mkdir(self.logs_dir)

    #def __str__(self): return self.models_dir


    def _mkdir(self, path):
        """Create required model directories

        :param path: string, path name
        """
        try:
            os.makedirs(path)
        except OSError as exc: #Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                raise

    def _run_identifier(self, reference_dir):
        # Retrieve run identifier
        run_id = 0
        for e in os.listdir(reference_dir):
            if e[:3] == 'run':
                r = int(e[3:])
                if r > run_id:
                    run_id = r

        run_id += 1

        return run_id

def init_tf_ops(sess):
    #Adapted from yadlt/utils/tf_utils.py
    """Initialize TensorFlow operations.
    This function initialize the following tensorflow ops:
        * init variables ops
        * summary ops
        * create model saver
    Parameters
    ----------
    sess : object
        Tensorflow `Session` object
    Returns
    -------
    tuple : (summary_merged, summary_writer)
        * tf merged summaries object
        * tf summary writer object
        * tf saver object
    """
    summary_merged = tf.summary.merge_all()
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    sess.run(init_op)

    run_id = Config()._run_identifier(Config().logs_dir)

    run_dir = os.path.join(Config().logs_dir, 'run' + str(run_id))
    print('Tensorboard logs dir for this run is %s' % (run_dir))

    summary_writer = tf.summary.FileWriter(run_dir, sess.graph)

    return (summary_merged, summary_writer, saver)
