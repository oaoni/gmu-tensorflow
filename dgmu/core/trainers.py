"""Training module"""

import tensorflow as tf

class Trainer(object):
    """Tensorflow optimizers wrapper"""

    def __init__(self, optimizer, lr):
        """Constructor

        :param optimizer: string, ['adam']
        :param lr: learning rate (float, default = 0.001)
        """

        assert optimizer in ['adam']

        if optimizer == 'adam':
            self.optimizer_ = tf.train.AdamOptimizer(lr)

    def compile(self, loss, name_scope = 'train'):
        """"Compile the optimizer

        :param loss: Tensor containing value to minimize
        :param name_scope: string, Name of scope for the optimizer grapher ops
        """

        with tf.name_scope(name_scope):
            return self.optimizer_.minimize(loss)


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


        if self.cost_function == 'softmax_cross_entropy':
            with tf.name_scope(self.cost_function):
                cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = y, labels = y_))
        else:
            cost = None

        if cost is not None:
            loss = (cost + reg_term) if reg_term is not None else (cost)
        else:
            loss = None

        if self.summary:
            tf.summary.scalar(self.cost_function, loss)

        return loss
