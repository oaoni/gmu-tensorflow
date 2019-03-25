"""Basic model class methods and skeleton"""

from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from dgmu.core import Borg, Config

class Model(object):
    """Generic abstract model in tensorflow"""

    def __init__(self, name = 'gmu'):
        """Constructor

        :param name: string, name of the model and also the model filename

        """

        self.name = name
        self.runid = Config()._run_identifier(Config().logs_dir)
        self.model_path = os.path.join(Config().models_dir, 'run' + str(self.runid), self.name+'.ckpt')
        self.logs_path = Config().logs_dir
        self.mod1 = None
        self.mod2 = None
        self.dropout_rate = None
        self.layer_nodes = []
        self.train_step = None
        self.cost = None

        #Tensorflow objects
        self.tf_graph = tf.Graph()
        self.tf_session = None
        self.tf_saver = None
        self.tf_merged_summaries = None
        self.tf_train_writer = None
        self.tf_test_writer = None


    def pretrain_procedure(self, layer_obj, layer_graphs, set_params_func,
                            train_mod1, train_mod2, validation_set = None):
        """Perform supervised pretraining of the representation network

        #rename layer_obj to representation_obj / rep_obj
        :param layer_obj: layer model
        :param layer_graphs: list of model tf.Graph objects
        :param set_params_func: function for setting parameters after pretraining
        :param train_mod1: Training modality 1
        :param train_mod2: training modality 2
        :param validation_set: validation set
        :param graph:
        :return: return data encoded by representation network
        """

        print('Supervised pretraining of the decision network...')

        layer_obj.fit(train_mod1, train_mod2, validation_set, graph = graph)

        train_encode = layer_obj.encode(train_mod1, train_mod2)

        return train_encode

    def _pretrain_gated_unit_and_gen_feed():
        """Supervised pretraining of the gated multimodal unit"""


        return None

    def finetune_procedure(self):
        """Performs supervised finetuning on the combined representation and decision networks

        """

        #with graph.as_default():

            #Set the paramters of the representation network after pretraining

        return None
