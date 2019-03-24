"""Gated multimodal unit based fusion and classification using Tensorflow """

from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from dgmu.core import Trainer, Layers, Score, Loss
from dgmu.core import SupervisedModel
from dgmu.core import utils

class GMU(SupervisedModel):
    """Biomodal GMU using Tensorflow"""

    def __init__(self, name = 'gmu', epochs = 1000, batch_size = 100,
                 cost_func = 'softmax_cross_entropy', lr = 0.01,
                 hidden_dim = 500, input_dr = 0, optimizer = 'adam',
                 reg_type = 'l2', beta = 0, n_features = 500, save_step = 100):
        """Constructor"""
        SupervisedModel.__init__(self, name, n_features)

        self.epochs = epochs
        self.batch_size = batch_size
        self.cost_func = cost_func
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.input_dr = input_dr
        self.optimizer = optimizer
        self.reg_type = reg_type
        self.beta = np.float32(beta)
        self.save_step = save_step

        self.loss = Loss(cost_func)
        self.trainer = Trainer(optimizer, lr)

        #Model variables
        self.W_h1 = None
        self.b_h1 = None
        self.W_h2 = None
        self.b_h2 = None
        self.W_y = None
        self.b_y = None

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
        self.y, self.W_y, self.b_y = Layers.linear(g_unit, n_classes, act_func = 'linear', name = 'output')

        variables = [self.W_h1, self.b_h1, self.W_h2, self.b_h2]
        reg_term = Layers.regularize(variables, self.reg_type, self.beta)

        self.cost = self.loss.compute(self.y, self.y_, reg_term = reg_term)
        self.train_step = self.trainer.compile(self.cost)

        #Accuracy
        self.accuracy = Score.accuracy(self.y, self.y_)

    def _create_placeholders(self, n_features, n_classes):
        """Creates the tensorflow placeholders for the model"""

        self.mod1 = tf.placeholder(tf.float32, [None, n_features], name = 'mod1')
        self.mod2 = tf.placeholder(tf.float32, [None, n_features], name = 'mod2')
        self.y_ = tf.placeholder(tf.float32, [None, n_classes], name = 'y-label')

    def _g_unit(self, mod1, mod2):

        mod3 = tf.concat([mod1, mod2], 1)

        h1, self.W_h1, self.b_h1 =  Layers.linear(mod1, self.hidden_dim, self.input_dr, name = 'modality1')

        h2, self.W_h2, self.b_h2 = Layers.linear(mod2, self.hidden_dim, self.input_dr, name = 'modality2')

        z, self.W_z, self.b_z = Layers.linear(mod3, self.hidden_dim, self.input_dr, name = 'modality3')

        h = tf.multiply(z, h1) + tf.multiply(tf.subtract(1.0, z), h2)

        return h

    def _train_model(self, trainX, trainY, valX, valY):
        """Train the model

        :param trainX: Training data, array_like, shape (n_samples, n_features)
        :param trainY: Training labels, array_like, shape (n_samples, n_classes)
        :param valX: Validation data, array_like, shape (N, n_features), (default = None)
        :param valY: Validation labels, array_like, shape (N, n_features), (default = None)
        :return self: trained model instance
        """

        pbar = tqdm(range(self.epochs))
        for epoch in pbar:
            self._run_train_step(trainX, trainY)

            #Measure cost and accuracy
            if (epoch % self.save_step == 0) and (valX is not None):

                #Split data by modality
                trainMod1 = trainX[:,:self.n_features]
                trainMod2 = trainX[:,self.n_features:]
                valMod1 = valX[:,:self.n_features]
                valMod2 = valX[:,self.n_features:]

                #Generate feed dictionaries
                trainFeed = {self.mod1:trainMod1, self.mod2:trainMod2, self.y_:trainY}
                valFeed = {self.mod1:valMod1, self.mod2:valMod2, self.y_:valY}

                #Compute cost
                cost = self.tf_session.run(self.cost, trainFeed)

                #Compute classification agreement
                #train_agreement = self.tf_session.run(self.accuracy, feed_dict = trainFeed)
                #test_agreement = self.tf_session.run(self.accuracy, feed_dict = valFeed)

                #print('epoch : ', epoch, 'cost : ', cost, 'train acc. :', test_agreement)

                #Add summary
                s = self.tf_session.run(self.tf_merged_summaries, feed_dict = trainFeed)
                self.tf_summary_writer.add_summary(s, epoch)

        return self

    def _run_train_step(self, trainX, trainY):
        """Run a training step

        :param trainX: Training data, array_like, shape (n_samples, n_features)
        :param trainY: Training labels, array_like, shape (n_samples, n_classes)
        """

        #Split data by modality
        mod1 = trainX[:,:self.n_features]
        mod2 = trainX[:,self.n_features:]

        batch_mod1, batch_mod2, batch_y_ = utils.get_batch(mod1, mod2, trainY, self.batch_size)
        feed = {self.mod1:batch_mod1, self.mod2:batch_mod2, self.y_:batch_y_}
        self.tf_session.run(self.train_step, feed_dict = feed)

        return self
