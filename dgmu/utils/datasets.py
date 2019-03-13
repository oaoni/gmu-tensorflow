"""Provide utility for loading sequence data"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import os

def load_cnv_rna_dcg_dataset(data_dir, mode = 'supervised', one_hot = True):
    """"Loads the cnv and rna sequence deeply connected features at
    dimensionality of 500 from the sequence_data directory.

    :param mode: 'supervised' or 'unsupervised' mode
    :param one_hot: True for one hot encoded labels
    :return train, validation, test data:
        for (X, y) if 'supervised',
        for (X) if 'unsupervised'
    """

    data = np.loadtxt(os.path.join(data_dir,'rna_DCF200.csv'), delimiter=',')

    return data
