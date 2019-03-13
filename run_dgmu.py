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

        data = datasets.load_cnv_rna_dcg_dataset(FLAGS.default_data_dir, mode='supervised', one_hot=True)
        print(data.shape)

    if FLAGS.dataset == 'custom':

        def load_from_np(custom_data_dir):
            if dataset_path != '':
                return np.loadtxt(custom_data_dir)
            else:
                return None

        X_train = load_from_np(FLAGS.train_dataset)
        X_val = load_from_np(FLAGS.valid_dataset)
        X_train = load_from_np(FLAGS.test_dataset)

    print("So far so good!")
