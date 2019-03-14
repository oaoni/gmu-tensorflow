"""dGMU core utilities module"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

def get_batch(X1, X2, Y_, size):
    """ Batch generator

    :param X1:
    :param X2:
    :param Y_:
    :return:
    """
    a = np.random.choice(len(X1), size, replace=False)
    return X1[a], X2[a], Y_[a]
