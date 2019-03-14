import numpy as np
import pandas as pd
import tensorflow as tf

from dgmu.models import gmu, deep_gmu
from dgmu.utils import datasets, utilities

# Define flags
flags = tf.app.flags
FLAGS = flags.FLAGS

#Gloabl configuration
flags.DEFINE_string('dataset', 'default', 'The data set to use. ["default", "custom"]')
flags.DEFINE_string('default_data_dir', 'sequence_data', 'Path to default directory with sequence data.')
flags.DEFINE_string('train_data_dir', '', 'Path to custom training set')
flags.DEFINE_string('valid_data_dir', '', 'Path to custom validation set.')
flags.DEFINE_string('test_data_dir', '', 'Path to custom test set.')

#dGMU specific configurations

#Define assertions
assert FLAGS.dataset in ['default', 'custom']
assert FLAGS.train_data_dir != '' if FLAGS.dataset == 'custom' else True

if __name__ == "__main__":

    if FLAGS.dataset == 'default':

        X_train, X_test, y_train, y_test = datasets.load_cnv_rna_dataset(FLAGS.default_data_dir, mode='supervised', one_hot=True)
        print(X_train.shape)
        print(y_train.shape)
        print(X_test.shape)
        print(y_test.shape)

    if FLAGS.dataset == 'custom':

        def load_custom(custom_data_dir, delimiter = ','):
            if dataset_path != '':
                return np.loadtxt(custom_data_dir, delimiter=',')
            else:
                return None

        X_train = load_custom(FLAGS.train_data_dir, delimiter=',')
        X_val = load_custom(FLAGS.valid_data_dir, delimiter=',')
        X_train = load_custom(FLAGS.test_data_dir, delimiter=',')

    print("So far so good!")
