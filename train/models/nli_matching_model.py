import tensorflow as tf
from nlp_blocks.transformer.funcs import gelu
from nlp_blocks.transformer.layers import LayerNormalization
from nlp_blocks.nac import NAC
from nlp_blocks.nalu import NALU

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
        self.in_drop = self.config['classifier_network']['in_dropout']
        self.h_drop = self.config['classifier_network']['dropout']
        self.h_activation = (self.config['classifier_network']
                                        ['hidden_activation'])
        self.h_ln = (self.config['classifier_network']['hidden_layer_norm'])
        self.kernel_initializer = (self.config['classifier_network']
                                              ['kernel_initializer'])
        self.kernel_constraint = (self.config['classifier_network']
                                             ['kernel_constraint'])
        self.gated = self.config['classifier_network']['gated']
        self.shortcut = self.config['classifier_network']['shortcut']
        if self.h_activation == 'gelu':
            self.h_activation = gelu
        elif self.h_activation == 'mish':
            self.h_activation = mish

        self.nac = False
        self.nalu = False

        if self.config['classifier_network']['hidden_layer_type'] == 'nac':
            self.nac = True
        elif self.config['classifier_network']['hidden_layer_type'] == 'nalu':
            self.nalu = True

        self.pred_activation = (self.config['classifier_network']
                                           ['prediction_activation'])

        if self.config['classifier_network']['hidden_layer_type'] == 'dense':
            self.fc_l1 = tf.keras.layers.Dense(self.h_dim,
                                               kernel_initializer=self.kernel_initializer,
                                               kernel_constraint=self.kernel_constraint,
                                               activation=None)
        elif self.nac:
            self.fc_l1 = NAC(self.h_dim)
        elif self.nalu:
            self.fc_l1 = NALU(self.h_dim, use_gating=False)

        if self.h_ln:
            self.fc_l1_ln = LayerNormalization(1e-5)
        self.fc_l1_act = self.h_activation
        self.in_drop = tf.keras.layers.Dropout(self.in_drop)
        self.fc_drop = tf.keras.layers.Dropout(self.h_drop)
        if self.config['classifier_network']['prediction_layer_type'] == 'dense':
            self.prediction = tf.keras.layers.Dense(
                self.config['classifier_network']['num_classes'],
                kernel_initializer=self.kernel_initializer,
                kernel_constraint=self.kernel_constraint,
                activation=self.pred_activation)
        elif self.nac:
            self.prediction = NAC(self.config['classifier_network']['num_classes'])
        elif self.nalu:
            self.prediction = NALU(self.config['classifier_network']['num_classes'],
                                   use_gating=False)

        if self.gated:
            self.gfc_l1 = tf.keras.layers.Dense(self.h_dim,
                                                kernel_initializer=self.kernel_initializer,
                                                kernel_constraint=self.kernel_constraint,
                                                activation=tf.keras.backend.sigmoid)

    def call(self, sent1, sent2):
        x_vec = tf.keras.backend.concatenate(
            [sent1, sent2, tf.math.abs(sent1 - sent2), sent1 * sent2], axis=1)
        x_vec = self.in_drop(x_vec)
        x = self.fc_l1(x_vec)
        if self.h_ln:
            x = self.fc_l1_ln(x)
        if self.gated:
            x = x*self.gfc_l1(x_vec)
        else:
            x = self.fc_l1_act(x)
        if self.shortcut:
            x = tf.keras.backend.concatenate([x_vec, x], axis=1)
        x = self.fc_drop(x)
        prediction = self.prediction(x)
        if self.nac or self.nalu:
            prediction = self.pred_activation(prediction)

        print("prediction:\t" + str(prediction))
        return prediction
