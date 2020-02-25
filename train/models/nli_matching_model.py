import tensorflow as tf
from nlp_blocks.transformer.funcs import gelu
from nlp_blocks.transformer.layers import LayerNormalization


def mish(inputs):
    return inputs * tf.keras.activations.tanh(
        tf.keras.activations.softplus(inputs))


class NliClassifierModel(tf.keras.Model):
    def __init__(self, config):
        super(NliClassifierModel, self).__init__()
        self.config = config

        self.s2v_dim = self.config['s2v_dim']

        self.h_dim = self.config['classifier_network']['hidden_dim']
        self.h_layer_cnt = (self.config['classifier_network']
                                       ['hidden_layer_cnt'])
        self.h_drop = self.config['classifier_network']['dropout']
        self.h_activation = (self.config['classifier_network']
                                        ['hidden_activation'])
        self.h_ln = (self.config['classifier_network']['hidden_layer_norm'])
        self.kernel_initializer = (self.config['classifier_network']
                                              ['kernel_initializer'])
        self.kernel_constraint = (self.config['classifier_network']
                                             ['kernel_constraint'])
        if self.h_activation == 'gelu':
            self.h_activation = gelu
        elif self.h_activation == 'mish':
            self.h_activation = mish

        self.pred_activation = (self.config['classifier_network']
                                           ['prediction_activation'])

        self.fc_l1 = tf.keras.layers.Dense(self.h_dim,
                                           kernel_initializer=self.kernel_initializer,
                                           kernel_constraint=self.kernel_constraint,
                                           activation=None)
        if self.h_ln:
            self.fc_l1_ln = LayerNormalization(1e-5)
        self.fc_l1_act = self.h_activation
        self.fc_drop = tf.keras.layers.Dropout(self.h_drop)
        self.prediction = tf.keras.layers.Dense(
            self.config['classifier_network']['num_classes'],
            kernel_initializer=self.kernel_initializer,
            kernel_constraint=self.kernel_constraint,
            activation=self.pred_activation)

    def call(self, sent1, sent2):
        x_vec = tf.keras.backend.concatenate(
            [sent1, sent2, tf.math.abs(sent1 - sent2), sent1 * sent2], axis=1)
        x = self.fc_l1(x_vec)
        if self.h_ln:
            x = self.fc_l1_ln(x)
        x = self.fc_l1_act(x)
        x = self.fc_drop(x)
        prediction = self.prediction(x)

        print("prediction:\t" + str(prediction))
        return prediction
