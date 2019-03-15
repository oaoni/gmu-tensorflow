"""Training module"""

import tensorflow as tf

class Loss(object):
    """Collection of cost functions"""

    def __init__(self, cost_function, summary = True, name = 'loss'):
        """Constructor

        :param cost_function: string, cost function ['softmax_cross_entropy']
        :param summary: bool, If true, attaches tf scalar summary to op
        :param name: string, name of scope
        """

        assert cost_function in ['softmax_cross_entropy','softmax_cross_entropy_reduce']

        self.cost_function = cost_function
        self.summary = summary
        self.name = name

    def compute(self, y, y_, reg_term = None):
        """Computes the loss equation tensor

        :param y: tf.Tensor Current label prediction
        :param y_: tf.Tensor True labels
        :param reg_term: tf.Tensor, regularization term tensor
        :return loss: tf.Tensor, loss function tensor
        """

        with tf.name_scope(self.name):
            if self.cost_function == 'softmax_cross_entropy':
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits = y, labels = y_)
            elif self.cost_function == 'softmax_cross_entropy_reduce':
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = y, labels = y_))
            else:
                loss = None

        if cost is not None:
            loss = (cost + reg_term) if reg_term is not None else (cost)
        else:
            loss = None

        return loss
