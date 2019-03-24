import numpy as np
import pandas as pd
import tensorflow as tf

from dgmu.models import gmu
from dgmu.core import utils
from dgmu.utils import datasets

# Define flags
flags = tf.app.flags
FLAGS = flags.FLAGS

#Gloabl configuration
flags.DEFINE_string('dataset', 'default', 'The data set to use. ["default", "custom"]')
flags.DEFINE_string('default_data_dir', 'sequence_data', 'Path to default directory with sequence data.')
flags.DEFINE_string('train_data_dir', '', 'Path to custom training set')
flags.DEFINE_string('valid_data_dir', '', 'Path to custom validation set.')
flags.DEFINE_string('test_data_dir', '', 'Path to custom test set.')
flags.DEFINE_string('name', 'gmu', 'Model name.')

#GMU specific configurations
flags.DEFINE_integer('epochs', 1000, 'Number of epochs.')
flags.DEFINE_integer('hidden_dim', 200, 'Number of hidden units.')
flags.DEFINE_integer('batch_size', 100, 'Size of each mini-batch.')
flags.DEFINE_integer('n_features', 200, 'Number of features in each modality.')
flags.DEFINE_integer('save_step', 10, 'Number of iterations to save summaries to filewriter.')
flags.DEFINE_boolean('summary', True, 'Whether to write summaries to logdir')

flags.DEFINE_float('regcoef', 5e-4, 'Regularization parameter. If 0, no regularization.')
flags.DEFINE_string('corr_type', 'none', 'Type of input corruption. ["none", "masking", "salt_and_pepper"]')
flags.DEFINE_float('corr_frac', 0., 'Fraction of the input to corrupt.')
flags.DEFINE_string('enc_act_func', 'tanh', 'Activation function for the encoder. ["sigmoid", "tanh"]')
flags.DEFINE_string('dec_act_func', 'none', 'Activation function for the decoder. ["sigmoid", "tanh", "none"]')
flags.DEFINE_string('loss_func', 'mse', 'Loss function. ["mse" or "cross_entropy"]')
flags.DEFINE_string('opt', 'sgd', '["sgd", "adagrad", "momentum", "adam"]')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('momentum', 0.5, 'Momentum parameter.')

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

        #Create model object
        gmu = gmu.GMU(name=FLAGS.name, hidden_dim=FLAGS.hidden_dim, n_features=FLAGS.n_features,
        save_step = FLAGS.save_step)
        print('Object model path is: ', gmu.model_path)
        #Fit the model
        gmu.fit(X_train, y_train, X_test, y_test, summary = FLAGS.summary)

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

    #Create the dgmu object
    #gmu_model = gmu.GMU(
    #name=FLAGS.name, ...
    #)

    #Fit the model
    #gmu.fit(X_train, y_train, X_test, y_test)
