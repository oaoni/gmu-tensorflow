"""Provide utility for loading sequence data"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import os

def load_cnv_rna_dataset(data_dir, mode = 'supervised', one_hot = True):
    """"Loads the cnv and rna sequence deeply connected features at
    dimensionality of 500 from the sequence_data directory.

    :param mode: 'supervised' or 'unsupervised' mode
    :param one_hot: True for one hot encoded labels
    :return train, validation, test data:
        for (X, y) if 'supervised',
        for (X) if 'unsupervised'
    """

    cnv = np.loadtxt(os.path.join(data_dir,'cnv_DCF200.csv'), delimiter=',')
    rna = np.loadtxt(os.path.join(data_dir,'rna_DCF200.csv'), delimiter=',')
    y = pd.read_csv(os.path.join(data_dir,'y_exp4.csv'),sep=',')

    if one_hot == True:
        lb = preprocessing.LabelBinarizer()
        labels = lb.fit_transform(y['sample_type'])
    else:
        labels = y['sample_type']

    #Generate training and test set
    X_train, X_test, y_train, y_test = train_test_split(np.concatenate([cnv, rna],axis=1), labels, test_size=0.30, random_state=42)

    if mode == 'supervised':
        return X_train, X_test, y_train, y_test

    elif mode == 'unsupervised':
        return X_train, X_test
