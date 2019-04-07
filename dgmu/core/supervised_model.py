"""Supervised GMU model class methods and skeleton"""

from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os

from dgmu.core.model import Model
from dgmu.core import Borg, Config, Score, utils
from dgmu.core import utils

class SupervisedModel(Model):
    """Provides methodology for a supervised gmu model

    fit(): Runs model training procedure
    predict(): Predicts labels given the trained model
    score(): Scores the model (mmean accuracy)
    """

    def __init__(self, name, n_features):
        """Constructor"""
        Model.__init__(self, name)

        self.n_features = n_features

        self.y_ = None
        self.accuracy = None

    def fit(self, trainX, trainY, valX = None, valY = None, graph = None, summary = True):
        """Fits the model to the training data

        :param trainX: Training data, array_like, shape (n_samples, n_features)
        :param trainY: Training labels, array_like, shape (n_samples, n_classes)
        :param valX: Validation data, array_like, shape (N, n_features), (default = None)
        :param valY: Validation labels, array_like, shape (N, n_features), (default = None)
        :param graph: Tensorflow graph object, tf.Graph, (default = None)
        """
        n_class = trainY.shape[1]
        g = graph if graph is not None else self.tf_graph

        #Make model run_id directory
        Config()._mkdir(os.path.join(Config().models_dir, 'run' + str(self.runid)))

        with g.as_default():
            #Build model
            self.build_model(self.n_features, n_class)
            with tf.Session() as self.tf_session:
                #Initialize tf parameters
                summary_objs = utils.init_tf_ops(self.tf_session, summary)
                self.tf_merged_summaries = summary_objs[0]
                self.tf_train_writer = summary_objs[1]
                self.tf_test_writer = summary_objs[2]
                self.tf_saver = summary_objs[3]
                #Train model
                self._train_model(trainX, trainY, valX, valY, summary)
                if summary:
                    print('Summaries added for training and test set.')
                    print("run Tensorboard --logdir='%s' to see results." % os.path.join(Config().logs_dir, 'run' + str(self.runid)))
                self.tf_saver.save(self.tf_session, self.model_path)
                print('Model saved to: ', self.model_path)

    def predict(self, testX, predict_proba = True, model_path = ''):
        """Predicts the labels for a test set example

        :param testX: Test data, array_like, shape (n_samples, n_features)
        :param predict_proba: boolean, if True, returns the class probabilities of each test example
        :param model_path: model path
        :return y: Predicted labels, shape
        """

        if model_path:
            self._restore_model(model_path)

        with self.tf_graph.as_default():
            with tf.Session() as self.tf_session:

                self.tf_saver.restore(self.tf_session, self.model_path)
                feed = {
                    self.mod1: testX[:,:self.n_features],
                    self.mod2: testX[:,self.n_features:]
                }

                if not predict_proba:
                    return np.argmax(self.y.eval(feed), 1)
                else:
                    return utils.softmax(self.y.eval(feed),axis=1)

    def score(self, testX, testY, model_path = ''):
        """Computes model score (mean accuracy)

        :param testX: Test data, array_like, shape (n_samples, n_features)
        :param testY: Test labels, array_like,shape (n_samples, n_features)
        :param model_path: string, model path, default ''
        :return accuracy: mean accuracy
        """

        if model_path:
            self._restore_model(model_path)

        with self.tf_graph.as_default():
            with tf.Session() as self.tf_session:

                self.tf_saver.restore(self.tf_session, self.model_path)
                feed = {
                self.mod1: testX[:,:self.n_features],
                self.mod2: testX[:,self.n_features:],
                self.y_: testY
                }

                return self.accuracy.eval(feed)

    def _restore_model(self, model_path):
        """Restores a supervised model from disk

        :param model_path: string, model path, default None
        """
        self.model_path = model_path
        self.tf_saver = tf.train.import_meta_graph(self.model_path + '.meta')

        self.tf_graph = tf.get_default_graph()
        self.mod1 = self.tf_graph.get_tensor_by_name('mod1:0')
        self.mod2 = self.tf_graph.get_tensor_by_name('mod2:0')
        self.y = self.tf_graph.get_tensor_by_name('output/Add:0')
        self.y_ = self.tf_graph.get_tensor_by_name('y_label:0')
        self.accuracy = self.tf_graph.get_tensor_by_name('accuracy/accuracy_op:0')

        print('Model restored from: ', self.model_path)
