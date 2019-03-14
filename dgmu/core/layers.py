"""Neural network layer operations"""

##################################################################
#Note. todo: consolodate the input dropout and hidden dropout methodologies.
#Compare with the methodologies for activation functions
##################################################################

import tensorflow

class Layers(object):
    """Implementation of NN layers"""

    @staticmethod
    def activate(self, linear, act_type):
        """Implements desired non linearity

        :param linear: tf.Tensor
        :param act_type: string, activation type ['sigmoid', softmax, linear, tanh, relu]
        """
        if name == 'sigmoid':
            return tf.nn.sigmoid(linear, name='sigmoid activation')
        elif name == 'softmax':
            return tf.nn.softmax(linear, name='softmax activation')
        elif name == 'linear':
            return linear
        elif name == 'tanh':
            return tf.nn.tanh(linear, name='tanh activation')
        elif name == 'relu':
            return tf.nn.relu(linear, name='relu activation')

    @staticmethod
    def linear(input_layer, hidden_dim, dropout_rate, name = 'linear'):
        """Creates fully connected layer

        :param input_layer: tf.Tensor (n_samples, n_features), input layer
        :param hidden_dim: int, hidden dimension size
        :param dropout rate: scalar [0,1), probability of an element being discarded
        :param name: string, name of scope
        :return (hidden_layer, W, b): tuple(tf.Tensor: hidden layer
                                            tf.Tensor: weights variable
                                            tf.Tensor: biases variable)
        """

        with tf.name_scope(name):
            input_dim = input_layer.get_shape()[1].value
            W = tf.Variable(tf.truncated_normal([input_dim, hidden_dim]))
            b = tf.Variable(tf.truncated_normal([hidden_dim]))
            _dropout = tf.nn.dropout(input_layer, rate = dropout_rate) #input dropout
            hidden_layer = tf.add(tf.matmul(_dropout, W), b)

            return (hidden_layer, W, b)

    @staticmethod
    def maxout(input_layer, hidden_dim, pool_depth, dropout_rate, name = 'maxout'):
        """Creates a fully connected maxout layer

        :param input_layer: tf.Tensor (n_samples, n_features), input layer
        :param hidden_dim: int, hidden dimension size
        :param dropout rate: scalar [0,1), probability of an element being discarded
        :param name: string, name of scope
        :param pool_depth: int, pooling depth for maxout operation
        :return (output_layer, W, b): tuple(tf.Tensor: output_layer
                                            tf.Tensor: weights variable
                                            tf.Tensor: biases variable)
        """

        with tf.name_scope(name):
            input_dim = input_layer.get_shape()[1].value
            W = tf.Variable(tf.truncated_normal([input_dim,(hidden_dim*pool_depth)]))
            b = tf.Variable(tf.truncated_normal([(hidden_dim*pool_depth)]))
            z = tf.add(tf.matmul(input_layer, W), b)
            _dropout = tf.nn.dropout(z, rate = dropout_rate) #hidden dropout
            _maxout = tf.reduce_max(tf.reshape(_dropout, [-1, hidden_dim, pool_depth]), axis=2)
            output_layer = tf.reshape(_maxout, [-1, hidden_dim])

            return (output_layer, W, b)

        @staticmethod
        def regularize(variables, reg_type, beta, name = 'regularization'):
            #Sampled from yadlt/core/layers.py
            """Computes regularization tensor

            :param variables: list of tf.Variable
            :param reg_type: string, type of regularization ['none','l1','l2']
            :param beta: float, regularization parameter
            :param name: string, name of scope
            :return tf.Tensor, regularization operation tensor
            """

            with tf.name_scope(name):
                if reg_type != 'none':
                    regs = tf.constant(0.0)
                    for v in variables:
                        if reg_type == 'l2':
                            regs = tf.add(regs, tf.nn.l2_loss(v))
                        elif reg_type == 'l1':
                            regs = tf.add(regs, tf.reduce_sum(tf.abs(v)))

                    return tf.multiply(beta, regs)

                else:

                    return None
