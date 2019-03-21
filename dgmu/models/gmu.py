"""Gated multimodal unit based fusion and classification using Tensorflow """

from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from dgmu.core import Trainer, Layers, Score, Loss
from dgmu.core import SupervisedModel
from dgmu.core import utils

class gmu(SupervisedModel):
    """Biomodal GMU using Tensorflow"""

    def __init__(self, name = 'gmu', epochs = 1000, batch_size = 100,
                 cost_func = 'softmax_cross_entropy', lr = 0.01,
                 hidden_dim = 500, input_dr = 0, optimizer = 'adam',
                 reg_type = 'l2', beta = 0, n_features = 500):
        """Constructor"""
        SupervisedModel.__init__(self, name)

        self.epochs = epochs
        self.batch_size = batch_size
        self.cost_func = cost_func
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.input_dr = input_dr
        self.optimizer = optimizer
        self.reg_type = reg_type
        self.beta = beta
        self.n_features = n_features

        self.loss = Loss(cost_func)
        self.trainer = Trainer(optimizer, lr)

        # Computational graph nodes
        self.input_data = None
        self.input_labels = None

        self.W_ = None
        self.b_ = None

        self.accuracy = None

    def build_model(self, n_features, n_classes):
        """Create the computational graph

        :param n_features: number of features
        :param n_classes: number of classes
        :return:
        """

        self._create_placeholders(n_features, n_classes)
        #self._create_variables(n_features, n_classes)
        #_g_unit() creates the variables

        g_unit = self._g_unit(self.mod1, self.mod2)
        self.y = Layers.linear(g_unit, n_classes, name = 'output')

        variables = [self.W_h1, self.b_h1, self.W_h2, self.b_h2]
        reg_term = Layers.regularize(variables, self.reg_type, self.beta)

        self.cost = self.loss.compute(self.y, self.y_, reg_term = reg_term)
        self.train_step = self.trainer.compile(self.cost)

    def _create_placeholders(n_features, n_classes):
        """Creates the tensorflow placeholders for the model"""

        self.mod1 = tf.placeholder(tf.float32, [None, n_features], name = 'mod1')
        self.mod2 = tf.placeholder(tf.float32, [None, n_features], name = 'mod1')
        self.y_ = tf.placeholder(tf.float32, [None, n_classes], name = 'y-label')

    def _g_unit(self, mod1, mod2):

        mod3 = tf.concat([mod1, mod2], 1)

        h1, W_h1, b_h1 =  Layers.linear(x1, self.hidden_dim, self.input_dr, name = 'modality1')
        h1 = Layers.activate(h1, 'relu')

        h2, W_h2, b_h2 = Layers.linear(x2, self.hidden_dim, self.input_dr, name = 'modality2')
        h2 = Layers.activate(h2, 'relu')

        z, W_z, b_z = Layers.linear(x3, self.hidden_dim, self.input_dr, name = 'modality3')
        h2 = Layers.activate(h2, 'relu')

        h = tf.multiply(z, h1) + tf.multiply(tf.subtract(1.0, z), h2)

        return h

    def _train_model(self):

        return self

    def _run_train_step(self):

        return self
