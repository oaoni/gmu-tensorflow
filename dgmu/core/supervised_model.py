"""Supervised GMU model class methods and skeleton"""

from __future__ import division
from __future__ import print_function

import tensorflow as tf

from dgmu.core.model import Model
from dgmu.core import utils

class SupervisedModel(Model):
    """Provides methodology for a supervised gmu model

    fit(): Runs model training procedure
    predict(): Predicts labels given the trained model
    score(): Scores the model (mmean accuracy)
    """

    def __init__(self, name):
        """Constructor"""
        Model.__init__(self, name)

    def fit(self, trainX, trainY, valX = None, valY = None, graph = None):
        """Fits the model to the training data

        :param trainX: Training data, array_like, shape (n_samples, n_features)
        :param trainY: Training labels, array_like, shape (n_samples, n_classes)
        :param valX: Validation data, array_like, shape (N, n_features), (default = None)
        :param valY: Validation labels, array_like, shape (N, n_features), (default = None)
        :param graph: Tensorflow graph object, tf.Graph, (default = None)
        """
        n_class = trainY.shape[1]
        g = graph if graph is not None else self.tf_graph

        with g.as_default():
            #Build model
            self.build_model(trainX.shape[1], n_class)
            with tf.Session() as self.tf_session:
                #Initialize tf parameters
                summary_objs = utils.init_tf_ops(self.tf_session)
                self.tf_merged_summaries = summary_objs[0]
                self.tf_summary_writer = summary_objs[1]
                self.tf_saver = summary_objs[2]
                #Train model
                self._train_model(train_X, train_Y, val_X, val_Y)
                # Save model
                self.tf_saver.save(self.tf_session, self.model_path)
