"""Multi-layer perceptron classification using Tensorflow"""

from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from dgmu.core import Trainer, Layers, Score, Loss
from dgmu.core import SupervisedModel
from dgmu.core import utils

class MLP(SupervisedModel):
    """Bimodal GMU using Tensorflow"""

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
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
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

        h1, self.W1, self.b1 =  Layers.linear(self.X, self.hidden_dim, self.input_dr, name = 'hidden1')

        h2, self.W2, self.b2 = Layers.linear(h1, self.hidden_dim, self.input_dr, name = 'hidden2')

        self.y, self.W_y, self.b_y = Layers.linear(h2, n_classes, act_func = 'linear', name = 'output')


        variables = [self.W1, self.b1, self.W2, self.b2]
        reg_term = Layers.regularize(variables, self.reg_type, self.beta)

        self.cost = self.loss.compute(self.y, self.y_, reg_term = reg_term)
        self.train_step = self.trainer.compile(self.cost)

        #Accuracy
        self.accuracy = Score.accuracy(self.y, self.y_)

    def _create_placeholders(self, n_features, n_classes):
        """Creates the tensorflow placeholders for the model"""

        self.X = tf.placeholder(tf.float32, [None, n_features], name = 'data')
        self.y_ = tf.placeholder(tf.float32, [None, n_classes], name = 'y_label')

    def _train_model(self, trainX, trainY, valX, valY, summary = True):
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
                #trainMod1 = trainX[:,:self.n_features]
                #trainMod2 = trainX[:,self.n_features:]
                #valMod1 = valX[:,:self.n_features]
                #valMod2 = valX[:,self.n_features:]

                #Generate feed dictionaries
                trainFeed = {self.X:trainX, self.y_:trainY}
                valFeed = {self.X:valX, self.y_:valY}

                #Compute cost
                cost = self.tf_session.run(self.cost, trainFeed)

                #Compute classification scores and record summaries
                s_train, trainScore = self.tf_session.run([self.tf_merged_summaries, self.accuracy], feed_dict = trainFeed)
                s_test, testScore = self.tf_session.run([self.tf_merged_summaries, self.accuracy], feed_dict = valFeed)

                pbar.set_description("Cost: %.4e Train Acc: %.5f Test Acc: %.5f" % (cost, trainScore, testScore))
                #print('epoch : ', epoch, 'cost : ', cost, 'train acc. :', trainScore, 'test acc. :', testScore)

                #Add summary
                if summary:
                    self.tf_train_writer.add_summary(s_train, epoch)
                    self.tf_test_writer.add_summary(s_test, epoch)

        return self

    def _run_train_step(self, trainX, trainY):
        """Run a training step

        :param trainX: Training data, array_like, shape (n_samples, n_features)
        :param trainY: Training labels, array_like, shape (n_samples, n_classes)
        """

        #Split data by modality

        batch_X, batch_y_ = utils.get_batchV2(trainX, trainY, self.batch_size)
        feed = {self.X:batch_X, self.y_:batch_y_}
        self.tf_session.run(self.train_step, feed_dict = feed)

        return self
