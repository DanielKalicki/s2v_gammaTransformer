"""
BERT Transformer implementation based on based on
https://github.com/Separius/BERT-keras.

"""
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dropout, Input, Dense
import tensorflow as tf
# from tensorflow.keras.layers import (AlphaDropout, SpatialDropout1D,
#                                      GaussianDropout, GaussianNoise)
from nlp_blocks.transformer.layers import (MultiHeadAttention, Gelu,
                                           LayerNormalization)
# from nlp_blocks.transformer.position import AddPositionalEncoding
# from nlp_blocks.wrappers.weight_normalization import WeightNormalization
from nlp_blocks.transformer.funcs import gelu
from nlp_blocks.nac import NAC


class MultiHeadSelfAttention:
    """
    Multi Head Self Attention implementation used in Transformer architecture.
    """
    def __init__(self, n_state: int, n_head: int, attention_dropout: float,
                 use_attn_mask: bool, layer_id: int, neg_inf: float,
                 modifications={}) -> None:
        assert n_state % n_head == 0
        self.n_state = n_state
        self.act_after_mha = False
        self.out_proj = False
        self.hidd_layer = False
        self.hidd_act = False
        self.out_mha = False
        self.use_bias = False,
        self.kernel_initializer = 'glorot_uniform'
        self.kernel_initializer = None

        if ('use_bias' in modifications) and (modifications['use_bias']):
            self.use_bias = True
        if 'kernel_initializer' in modifications:
            self.kernel_initializer = modifications['kernel_initializer']
        if 'kernel_constraint' in modifications:
            self.kernel_constraint = modifications['kernel_constraint']

        mha_dim = n_state
        if 'inner_dim' in modifications:
            mha_dim = modifications['inner_dim']

        self.c_attn = Dense(3 * mha_dim, use_bias=self.use_bias,
                            kernel_initializer=self.kernel_initializer,
                            kernel_constraint=self.kernel_constraint,
                            name='layer_{}/c_attn'.format(layer_id))

        self.attn = MultiHeadAttention(n_head, mha_dim, attention_dropout,
                                       use_attn_mask, neg_inf,
                                       name='layer_{}/self_attention'
                                            .format(layer_id))

        if ('activation_after_mha' in modifications) \
           and (modifications['activation_after_mha'] is not None):
            self.act_after_mha = True
            if modifications['activation_after_mha'] == 'gelu':
                self.activation = gelu
            else:
                self.activation = modifications['activation_after_mha']

        if ('hidden_layer' in modifications) \
           and (modifications['hidden_layer']):
            self.hidd_layer = True
            self.hidd_dim = modifications['hidden_dim']
            self.c_attn_hidd = Dense(self.hidd_dim, use_bias=self.use_bias,
                                     kernel_initializer=self.kernel_initializer,
                                     kernel_constraint=self.kernel_constraint,
                                     name='layer_{}/c_attn_hidd'
                                          .format(layer_id))
            if ('hidden_activation' in modifications) \
               and (modifications['hidden_activation'] is not None):
                self.hidd_act = True
                if modifications['hidden_activation'] == 'gelu':
                    self.h_activation = gelu
                else:
                    self.h_activation = modifications['activation_after_mha']

        if ('output_mha' in modifications) \
           and (modifications['output_mha']):
            self.out_mha = True
            self.c_out_attn = Dense(3 * n_state, use_bias=self.use_bias,
                                    kernel_initializer=self.kernel_initializer,
                                    kernel_constraint=self.kernel_constraint,
                                    name='layer_{}/c_out_attn'
                                         .format(layer_id))
            self.out_attn = MultiHeadAttention((modifications
                                                ['output_mha_num_heads']),
                                               mha_dim, attention_dropout,
                                               use_attn_mask, neg_inf,
                                               name='layer_{}/out_attn'
                                                    .format(layer_id))

        if ('output_projection' in modifications) \
           and (modifications['output_projection']):
            self.out_proj = True
            self.c_attn_proj = Dense(n_state, use_bias=self.use_bias,
                                     kernel_initializer=self.kernel_initializer,
                                     kernel_constraint=self.kernel_constraint,
                                     name='layer_{}/c_attn_proj'
                                          .format(layer_id))

    def __call__(self, x, mask):
        output = self.c_attn(x)
        output = self.attn(output) if mask is None else \
            self.attn([output, mask])

        if self.act_after_mha:
            output = self.activation(output)

        if self.out_mha:
            output = self.c_out_attn(output)
            output = self.out_attn(output) if mask is None else \
                self.out_attn([output, mask])

        if self.hidd_layer:
           output = self.c_attn_hidd(output)
           if self.hidd_act:
               output = self.h_activation(output)

        if self.out_proj:
            output = self.c_attn_proj(output)
        return output


class PositionWiseFF:
    """
    Feedforward network implementation used in Transformer architecture.
    """
    def __init__(self, n_state: int, d_hid: int, layer_id: int,
                 accurate_gelu: bool, modifications={}) -> None:
        self.use_bias = False
        self.kernel_initializer = 'glorot_uniform'
        self.kernel_initializer = None
        self.kernel_constraint = None
        self.gated = False
        self.nac = False
        if ('use_bias' in modifications) and (modifications['use_bias']):
            self.use_bias = True
        if 'kernel_initializer' in modifications:
            self.kernel_initializer = modifications['kernel_initializer']
        if 'kernel_constraint' in modifications:
            self.kernel_constraint = modifications['kernel_constraint']
        if ('gated' in modifications) and (modifications['gated']):
            self.gated = True
        if ('nac' in modifications) and (modifications['nac']):
            self.nac = True

        if not self.nac:
            self.c_fc = Dense(d_hid, use_bias=self.use_bias,
                              kernel_initializer=self.kernel_initializer,
                              kernel_constraint=self.kernel_constraint,
                              name='layer_{}/c_fc'.format(layer_id))
        else:
            self.c_fc = NAC(d_hid)

        self.activation = Gelu(accurate=accurate_gelu,
                               name='layer_{}/gelu'.format(layer_id))

        self.c_ffn_proj = Dense(n_state, use_bias=self.use_bias,
                                    kernel_initializer=self.kernel_initializer,
                                    kernel_constraint=self.kernel_constraint,
                                    name='layer_{}/c_ffn_proj'.format(layer_id))
        if self.gated:
            self.sigmoid_act = tf.keras.activations.sigmoid
            self.g_fc = Dense(n_state, use_bias=self.use_bias,
                              kernel_initializer=self.kernel_initializer,
                              kernel_constraint=self.kernel_constraint,
                              name='layer_{}/g_fc'.format(layer_id))


    def __call__(self, x):
        output = self.c_fc(x)
        output = self.activation(output)
        output = self.c_ffn_proj(output)
        if self.gated:
            output = self.sigmoid_act(output)*self.g_fc(x)
        return output


class GatedEncoderLayer:
    def __init__(self, n_state: int, n_head: int, d_hid: int,
                 residual_dropout: float, attention_dropout: float,
                 use_attn_mask: bool, layer_id: int, neg_inf: float,
                 ln_epsilon: float, accurate_gelu: bool,
                 kernel_initializer, kernel_constraint,
                 normalization_position='post',
                 gate_type='None', mha_modifications={},
                 ffn_modifications={}) -> None:
        self.gate_type = gate_type
        self.ffn_layer = ffn_modifications['ffn_layer']
        self.kernel_initializer = kernel_initializer
        self.kernel_constraint = kernel_constraint
        self.normalization_position = normalization_position
        self.small_ffn = ffn_modifications['small_ffn']

        self.attention = MultiHeadSelfAttention(n_state, n_head,
                                                attention_dropout,
                                                use_attn_mask, layer_id,
                                                neg_inf, mha_modifications)
        self.drop1 = Dropout(residual_dropout,
                             name='layer_{}/ln_1_drop'.format(layer_id))
        self.ln1 = LayerNormalization(ln_epsilon,
                                      name='layer_{}/ln_1'.format(layer_id))

        if self.ffn_layer:
            self.ffn = PositionWiseFF(n_state, d_hid, layer_id, accurate_gelu,
                                      ffn_modifications)
            self.drop2 = Dropout(residual_dropout,
                                 name='layer_{}/ln_2_drop'.format(layer_id))
            self.ln2 = LayerNormalization(ln_epsilon,
                                          name='layer_{}/ln_2'
                                               .format(layer_id))
        if self.small_ffn:
            self.sffn = PositionWiseFF(n_state, n_state/2, layer_id+100, accurate_gelu,
                                       ffn_modifications)
            self.drop3 = Dropout(residual_dropout,
                                 name='layer_{}/ln_3_drop'.format(layer_id))
            self.ln3 = LayerNormalization(ln_epsilon,
                                          name='layer_{}/ln_3'.format(layer_id))


        if self.gate_type != 'None':
            self.gate1_dense = Dense(n_state, use_bias=True,
                                     kernel_initializer=self.kernel_initializer,
                                     kernel_constraint=self.kernel_constraint,
                                     activation=tf.keras.backend.sigmoid,
                                     name='layer_{}/gate1'.format(layer_id))
            if self.ffn_layer:
                self.gate2_dense = Dense(n_state, use_bias=True,
                                         kernel_initializer=self.kernel_initializer,
                                         kernel_constraint=self.kernel_constraint,
                                         activation=tf.keras.backend.sigmoid,
                                         name='layer_{}/gate2'
                                              .format(layer_id))
            if self.small_ffn:
                self.gate3_dense = Dense(n_state, use_bias=True,
                                         kernel_initializer=self.kernel_initializer,
                                         kernel_constraint=self.kernel_constraint,
                                         activation=tf.keras.backend.sigmoid,
                                         name='layer_{}/gate3'
                                              .format(layer_id))
            if self.gate_type == 'Wg(y)*tanh(Ug(y)) + x':
                self.gate1_Ug = Dense(n_state, use_bias=False,
                                      kernel_initializer=self.kernel_initializer,
                                      kernel_constraint=self.kernel_constraint,
                                      activation=tf.keras.backend.tanh,
                                      name='layer_{}/gate1_Ug'
                                           .format(layer_id))
                if self.ffn_layer:
                    self.gate2_Ug = Dense(n_state, use_bias=False,
                                          kernel_initializer=self.kernel_initializer,
                                          kernel_constraint=self.kernel_constraint,
                                          activation=tf.keras.backend.tanh,
                                          name='layer_{}/gate2_Ug'
                                               .format(layer_id))
            if self.gate_type == 'Ffn(x,y)*y + x':
                self.gate1h_dense = Dense(n_state*2, use_bias=True,
                                          kernel_initializer=self.kernel_initializer,
                                          kernel_constraint=self.kernel_constraint,
                                          activation=gelu,
                                          name='layer_{}/gate1h'.format(layer_id))
                if self.ffn_layer:
                    self.gate2h_dense = Dense(n_state*2, use_bias=True,
                                              kernel_initializer=self.kernel_initializer,
                                              kernel_constraint=self.kernel_constraint,
                                              activation=gelu,
                                              name='layer_{}/gate2h'
                                                   .format(layer_id))
                if self.small_ffn:
                    self.gate3h_dense = Dense(n_state/2, use_bias=True,
                                              kernel_initializer=self.kernel_initializer,
                                              kernel_constraint=self.kernel_constraint,
                                              activation=gelu,
                                              name='layer_{}/gate3h'
                                                   .format(layer_id))
            if self.gate_type == 'FfnNac(x,y)*y + x':
                self.gate1h_dense = NAC(n_state*2)
                if self.ffn_layer:
                    self.gate2h_dense = NAC(n_state*2)
                if self.small_ffn:
                    self.gate3h_dense = NAC(n_state/2)
            if self.gate_type == 'Mha(x,y)*y + x':
                self.gate_attn1 = Dense(3 * n_state, use_bias=False,
                                    kernel_initializer=self.kernel_initializer,
                                    kernel_constraint=self.kernel_constraint,
                                    name='layer_{}/c_attn_gate1'.format(layer_id))

                self.gate_mha1 = MultiHeadAttention(n_head, n_state, attention_dropout,
                                               use_attn_mask, neg_inf,
                                               name='layer_{}/self_attention_gate1'
                                                    .format(layer_id))
                if self.ffn_layer:
                    self.gate_attn2 = Dense(3 * n_state, use_bias=False,
                                        kernel_initializer=self.kernel_initializer,
                                        kernel_constraint=self.kernel_constraint,
                                        name='layer_{}/c_attn_gate2'.format(layer_id))

                    self.gate_mha2 = MultiHeadAttention(n_head, n_state, attention_dropout,
                                                   use_attn_mask, neg_inf,
                                                   name='layer_{}/self_attention_gate2'
                                                        .format(layer_id))
            if self.gate_type == 'STE(x,y)*y + x':
                self.gate1_ste1_dense = Dense(n_state, use_bias=True,
                                         kernel_initializer=self.kernel_initializer,
                                         kernel_constraint=self.kernel_constraint,
                                         activation=tf.keras.backend.sigmoid,
                                         name='layer_{}/gate1_ste1'.format(layer_id))
                self.gate1_ste2_dense = Dense(n_state, use_bias=True,
                                         kernel_initializer=self.kernel_initializer,
                                         kernel_constraint=self.kernel_constraint,
                                         activation=tf.keras.backend.sigmoid,
                                         name='layer_{}/gate1_ste2'.format(layer_id))
                self.gate1_ste3_dense = Dense(n_state, use_bias=True,
                                         kernel_initializer=self.kernel_initializer,
                                         kernel_constraint=self.kernel_constraint,
                                         activation=tf.keras.backend.sigmoid,
                                         name='layer_{}/gate1_ste3'.format(layer_id))
                self.drop1_ste0 = Dropout(0.1,
                                     name='layer_{}/ln_1_drop_ste0'.format(layer_id))
                self.drop1_ste1 = Dropout(0.1,
                                     name='layer_{}/ln_1_drop_ste1'.format(layer_id))
                self.drop1_ste2 = Dropout(0.1,
                                     name='layer_{}/ln_1_drop_ste2'.format(layer_id))
                self.drop1_ste3 = Dropout(0.1,
                                     name='layer_{}/ln_1_drop_ste3'.format(layer_id))
                if self.ffn_layer:
                    self.gate2_ste1_dense = Dense(n_state, use_bias=True,
                                             kernel_initializer=self.kernel_initializer,
                                             kernel_constraint=self.kernel_constraint,
                                             activation=tf.keras.backend.sigmoid,
                                             name='layer_{}/gate2_ste1'
                                                  .format(layer_id))
                    self.gate2_ste2_dense = Dense(n_state, use_bias=True,
                                             kernel_initializer=self.kernel_initializer,
                                             kernel_constraint=self.kernel_constraint,
                                             activation=tf.keras.backend.sigmoid,
                                             name='layer_{}/gate2_ste2'
                                                  .format(layer_id))
                    self.gate2_ste3_dense = Dense(n_state, use_bias=True,
                                             kernel_initializer=self.kernel_initializer,
                                             kernel_constraint=self.kernel_constraint,
                                             activation=tf.keras.backend.sigmoid,
                                             name='layer_{}/gate2_ste3'
                                                  .format(layer_id))
                    self.drop2_ste0 = Dropout(0.1,
                                         name='layer_{}/ln_2_drop_ste0'.format(layer_id))
                    self.drop2_ste1 = Dropout(0.1,
                                         name='layer_{}/ln_2_drop_ste1'.format(layer_id))
                    self.drop2_ste2 = Dropout(0.1,
                                         name='layer_{}/ln_2_drop_ste2'.format(layer_id))
                    self.drop2_ste3 = Dropout(0.1,
                                         name='layer_{}/ln_2_drop_ste3'.format(layer_id))

    def gate_output(self, x, y, g_dense, drop):
        if self.gate_type == 'Wg(x)*y + x':
            # g(x, y) = σ(Wg*x + b)*y + x
            g_var = x
        elif self.gate_type == 'Wg(x,y)*y + x':
            # g(x, y) = σ(Wg*(x||y) + b)*y + x
            g_var = tf.keras.backend.concatenate([x, y], axis=2)
        return drop(g_dense(g_var) * y) + x

    def gate_output_ffn(self, x, y, g_dense, g_dense_hid, drop):
        g_var = tf.keras.backend.concatenate([x, y], axis=2)
        return drop(g_dense(g_dense_hid(g_var)) * y) + x

    def gate_output_mha(self, x, y, g_dense, g_mha_att, g_mha, mask, drop):
        g_var = tf.keras.backend.concatenate([x, y], axis=2)
        return drop(g_dense(g_mha([g_mha_att(g_var), mask])) * y) + x

    def gate_output_ste(self, x, y, g_dense1, g_dense2, g_dense3, g_dense4,
                        drop, ste_drop1, ste_drop2, ste_drop3, ste_drop4):
        g_var = tf.keras.backend.concatenate([x, y], axis=2)

        ste1 = ste_drop1(g_dense1(g_var))
        ste2 = ste_drop2(g_dense2(g_var))
        ste3 = ste_drop3(g_dense3(g_var))
        ste4 = ste_drop4(g_dense4(g_var))
        ste = (ste1 + ste2 + ste3 + ste4)/4

        return drop(ste * y) + x

    def gate_sigmoid_tanh(self, x, y, g1_dense, g2_dense, drop):
        # g(x, y) = σ(Wg*y + b)*tanh(Ug*y) + x
        return drop(g1_dense(y)*g2_dense(y)) + x

    def gate_input(self, x, y, g_dense, drop):
        # g(x, y) = σ(Wg*y + b)*x + y
        return g_dense(x)*x + drop(y)

    def gate_highway(self, x, y, g_dense, drop):
        # g(x, y) = σ(Wg*x + b)*x + (1-σ(Wg*x + b))*y
        gate = g_dense(x)
        return gate*x + drop((1-gate)*y)

    def __call__(self, x, mask):
        if self.small_ffn:
            xSubLayer = x
            if (self.normalization_position == 'pre') or \
               (self.normalization_position == 'preMod'):
                xSubLayer = self.ln3(x)
            y = self.sffn(xSubLayer)

            if (self.gate_type == 'Ffn(x,y)*y + x') or \
               (self.gate_type == 'FfnNac(x,y)*y + x'):
                x = self.gate_output_ffn(x, y, self.gate3_dense, self.gate3h_dense,
                                         self.drop3)

            if self.normalization_position == 'post':
                x = self.ln3(x)

        if self.normalization_position == 'preMod':
            x = self.ln1(x)
        xSubLayer = x
        if self.normalization_position == 'pre':
            xSubLayer = self.ln1(x)
        y = self.attention(xSubLayer, mask)

        if self.gate_type in ['Wg(x)*y + x', 'Wg(x,y)*y + x']:
            x = self.gate_output(x, y, self.gate1_dense, self.drop1)
        elif (self.gate_type == 'Ffn(x,y)*y + x') or \
             (self.gate_type == 'FfnNac(x,y)*y + x'):
            x = self.gate_output_ffn(x, y, self.gate1_dense, self.gate1h_dense,
                                     self.drop1)
        elif self.gate_type == 'Mha(x,y)*y + x':
            x = self.gate_output_mha(x, y, self.gate1_dense, self.gate_attn1,
                                     self.gate_mha1, mask, self.drop1)
        elif self.gate_type == 'STE(x,y)*y + x':
            x = self.gate_output_ste(x, y, self.gate1_dense,
                                     self.gate1_ste1_dense,
                                     self.gate1_ste2_dense,
                                     self.gate1_ste3_dense,
                                     self.drop1,
                                     self.drop1_ste0, self.drop1_ste1,
                                     self.drop1_ste2, self.drop1_ste3)
        elif self.gate_type == 'Wg(y)*tanh(Ug(y)) + x':
            x = self.gate_sigmoid_tanh(x, y, self.gate1_dense, self.gate1_Ug,
                                       self.drop1)
        elif self.gate_type == 'Wg(x)*x + y':
            x = self.gate_input(x, y, self.gate1_dense, self.drop1)
        elif self.gate_type == 'Wg(x)*x + (1-Wg(x))*y':
            x = self.gate_highway(x, y, self.gate1_dense, self.drop1)
        elif self.gate_type == 'None':
            x = x + self.drop1(y)

        if self.normalization_position == 'post':
            x = self.ln1(x)

        if self.ffn_layer:
            xSubLayer = x
            if (self.normalization_position == 'pre') or \
               (self.normalization_position == 'preMod'):
                xSubLayer = self.ln2(x)
            y = self.ffn(xSubLayer)

            if self.gate_type in ['Wg(x)*y + x', 'Wg(x,y)*y + x']:
                x = self.gate_output(x, y, self.gate2_dense, self.drop2)
            elif (self.gate_type == 'Ffn(x,y)*y + x') or \
                 (self.gate_type == 'FfnNac(x,y)*y + x'):
                x = self.gate_output_ffn(x, y, self.gate2_dense, self.gate2h_dense,
                                         self.drop2)
            elif self.gate_type == 'Mha(x,y)*y + x':
                x = self.gate_output_mha(x, y, self.gate2_dense, self.gate_attn2,
                                         self.gate_mha2, mask, self.drop2)
            elif self.gate_type == 'STE(x,y)*y + x':
                x = self.gate_output_ste(x, y, self.gate2_dense,
                                         self.gate2_ste1_dense,
                                         self.gate2_ste2_dense,
                                         self.gate2_ste3_dense,
                                         self.drop2,
                                         self.drop2_ste0, self.drop2_ste1,
                                         self.drop2_ste2, self.drop2_ste3)
            elif self.gate_type == 'Wg(y)*tanh(Ug(y)) + x':
                x = self.gate_sigmoid_tanh(x, y, self.gate2_dense,
                                           self.gate2_Ug, self.drop2)
            elif self.gate_type == 'Wg(x)*x + y':
                x = self.gate_input(x, y, self.gate2_dense, self.drop2)
            elif self.gate_type == 'Wg(x)*x + (1-Wg(x))*y':
                x = self.gate_highway(x, y, self.gate2_dense, self.drop2)
            elif self.gate_type == 'None':
                x = x + self.drop2(y)

            if self.normalization_position == 'post':
                x = self.ln2(x)

        return x


def create_gated_transformer(embedding_dim: int = 768, max_len: int = 512,
                             num_heads: int = 12, num_layers: int = 12,
                             attention_dropout: float = 0.0,
                             d_hid: int = 768 * 4,
                             residual_dropout: float = 0.0,
                             use_attn_mask: bool = True, neg_inf: float = -1e9,
                             layer_norm_epsilon: float = 1e-5,
                             kernel_initializer='glorot_uniform',
                             kernel_constraint=None,
                             accurate_gelu: bool = True,
                             normalization_position='post',
                             gate_type='None',
                             mha_modifications={}, ffn_modifications={}
                             ) -> tensorflow.keras.Model:

    if gate_type not in ['None', 'Wg(x)*y + x', 'Wg(x,y)*y + x',
                         'Wg(y)*tanh(Ug(y)) + x', 'Wg(x)*x + y',
                         'Wg(x)*x + (1-Wg(x))*y', 'Ffn(x,y)*y + x',
                         'FfnNac(x,y)*y + x', 'STE(x,y)*y + x',
                         'Mha(x,y)*y + x']:
        raise ValueError('Unknown config.sentence_encoder.transformer.'
                         + 'gate_type.')

    input_ = Input(batch_shape=(None, max_len, embedding_dim),
                   name='input', dtype='float32')
    attn_mask = Input(batch_shape=(None, 1, max_len, max_len),
                      name='attention_mask_input', dtype=K.floatx()
                      ) if use_attn_mask else None
    inputs = [input_]
    x = input_
    for i in range(num_layers):
        x = GatedEncoderLayer(embedding_dim, num_heads, d_hid,
                              residual_dropout, attention_dropout,
                              use_attn_mask, i, neg_inf, layer_norm_epsilon,
                              accurate_gelu, kernel_initializer,
                              kernel_constraint,
                              normalization_position,
                              gate_type, mha_modifications,
                              ffn_modifications)(x, attn_mask)
    if use_attn_mask:
        inputs.append(attn_mask)
    return tensorflow.keras.Model(inputs=inputs, outputs=[x],
                                  name='GatedTransformer')


class MultiHeadSelfAttentionPool:
    """
    Modified Multi Head Attention used to expand the word embeddings before
        pooling.
    """
    def __init__(self, n_state: int, n_head: int, attention_dropout: float,
                 use_attn_mask: bool, layer_id: int, neg_inf: float,
                 output_projection: bool, output_dim: int,
                 input_ffn, input_ffn_dim: int, kernel_initializer,
                 kernel_constraint
                 ) -> None:
        assert n_state % n_head == 0
        self.n_state = n_state
        self.output_projection = output_projection
        self.input_ffn = input_ffn
        self.kernel_initializer = kernel_initializer
        self.kernel_constraint = kernel_constraint

        if 'NAC' not in input_ffn:
            self.c_att_q = Dense(n_state, use_bias=False, activation=None,
                                 kernel_initializer=self.kernel_initializer,
                                 kernel_constraint=self.kernel_constraint,
                                 name='layer_{}/c_att_q'.format(layer_id))
            self.c_att_k = Dense(n_state, use_bias=False, activation=None,
                                 kernel_initializer=self.kernel_initializer,
                                 kernel_constraint=self.kernel_constraint,
                                 name='layer_{}/c_att_k'.format(layer_id))
            self.c_att_v = Dense(n_state if 'mha' not in self.input_ffn
                                 else 3*n_state, use_bias=False, activation=None,
                                 kernel_initializer=self.kernel_initializer,
                                 kernel_constraint=self.kernel_constraint,
                                 name='layer_{}/c_att_v'.format(layer_id))
        else:
            self.c_att_q = NAC(n_state)
            self.c_att_k = NAC(n_state)
            self.c_att_v = NAC(n_state if 'mha' not in self.input_ffn else 3*n_state)

        if 'q' in input_ffn:
            self.c_att_q2 = Dense(input_ffn_dim, use_bias=False,
                                  activation=gelu,
                                  kernel_initializer=self.kernel_initializer,
                                  kernel_constraint=self.kernel_constraint,
                                  name='layer_{}/c_att_q2'.format(layer_id))
        if 'k' in input_ffn:
            self.c_att_k2 = Dense(input_ffn_dim, use_bias=False,
                                  activation=gelu,
                                  kernel_initializer=self.kernel_initializer,
                                  kernel_constraint=self.kernel_constraint,
                                  name='layer_{}/c_att_k2'.format(layer_id))
        if 'v' in input_ffn:
            self.c_att_v2 = Dense(input_ffn_dim, use_bias=False,
                                  activation=gelu,
                                  kernel_initializer=self.kernel_initializer,
                                  kernel_constraint=self.kernel_constraint,
                                  name='layer_{}/c_att_v2'.format(layer_id))
        if 'mha' in input_ffn:
            self.attn_v2 = MultiHeadAttention(n_head, n_state,
                                              attention_dropout,
                                              use_attn_mask, neg_inf,
                                              name='layer_{}/v_self_attention'
                                                   .format(layer_id))
        if 'STE' in input_ffn:
            if 'Q' in input_ffn:
                self.ste_drop_q1 = Dropout(0.1)
                self.ste_drop_q2 = Dropout(0.1)
                self.ste_drop_q3 = Dropout(0.1)
                self.ste_drop_q4 = Dropout(0.1)
                self.c_att_q2 = Dense(n_state, use_bias=False,
                                      activation=None,
                                      kernel_initializer=self.kernel_initializer,
                                      kernel_constraint=self.kernel_constraint,
                                      name='layer_{}/c_att_q2'.format(layer_id))
                self.c_att_q3 = Dense(n_state, use_bias=False,
                                      activation=None,
                                      kernel_initializer=self.kernel_initializer,
                                      kernel_constraint=self.kernel_constraint,
                                      name='layer_{}/c_att_q3'.format(layer_id))
                self.c_att_q4 = Dense(n_state, use_bias=False,
                                      activation=None,
                                      kernel_initializer=self.kernel_initializer,
                                      kernel_constraint=self.kernel_constraint,
                                      name='layer_{}/c_att_q4'.format(layer_id))
            if 'K' in input_ffn:
                self.ste_drop_k1 = Dropout(0.1)
                self.ste_drop_k2 = Dropout(0.1)
                self.ste_drop_k3 = Dropout(0.1)
                self.ste_drop_k4 = Dropout(0.1)
                self.c_att_k2 = Dense(n_state, use_bias=False,
                                      activation=None,
                                      kernel_initializer=self.kernel_initializer,
                                      kernel_constraint=self.kernel_constraint,
                                      name='layer_{}/c_att_k2'.format(layer_id))
                self.c_att_k3 = Dense(n_state, use_bias=False,
                                      activation=None,
                                      kernel_initializer=self.kernel_initializer,
                                      kernel_constraint=self.kernel_constraint,
                                      name='layer_{}/c_att_k3'.format(layer_id))
                self.c_att_k4 = Dense(n_state, use_bias=False,
                                      activation=None,
                                      kernel_initializer=self.kernel_initializer,
                                      kernel_constraint=self.kernel_constraint,
                                      name='layer_{}/c_att_k4'.format(layer_id))
            if 'V' in input_ffn:
                self.ste_drop_v1 = Dropout(0.1)
                self.ste_drop_v2 = Dropout(0.1)
                self.ste_drop_v3 = Dropout(0.1)
                self.ste_drop_v4 = Dropout(0.1)
                self.c_att_v2 = Dense(n_state, use_bias=False,
                                      activation=None,
                                      kernel_initializer=self.kernel_initializer,
                                      kernel_constraint=self.kernel_constraint,
                                      name='layer_{}/c_att_v2'.format(layer_id))
                self.c_att_v3 = Dense(n_state, use_bias=False,
                                      activation=None,
                                      kernel_initializer=self.kernel_initializer,
                                      kernel_constraint=self.kernel_constraint,
                                      name='layer_{}/c_att_v3'.format(layer_id))
                self.c_att_v4 = Dense(n_state, use_bias=False,
                                      activation=None,
                                      kernel_initializer=self.kernel_initializer,
                                      kernel_constraint=self.kernel_constraint,
                                      name='layer_{}/c_att_v4'.format(layer_id))

        self.attn = MultiHeadAttention(n_head, n_state, attention_dropout,
                                       use_attn_mask, neg_inf,
                                       name='layer_{}/self_attention'
                                            .format(layer_id))
        if self.output_projection:
            self.c_attn_proj = Dense(output_dim, use_bias=False,
                                     kernel_initializer=self.kernel_initializer,
                                     kernel_constraint=self.kernel_constraint,
                                     name='layer_{}/c_attn_proj'
                                          .format(layer_id))

    def __call__(self, x, mask):
        if 'STE' not in self.input_ffn:
            q = self.c_att_q2(x) if 'q' in self.input_ffn else x
            k = self.c_att_k2(x) if 'k' in self.input_ffn else x
            v = self.c_att_v2(x) if 'v' in self.input_ffn else x
            v = self.attn_v2([self.c_att_v(v), mask]) if 'mha' in \
                self.input_ffn else v
            output = tf.keras.backend.concatenate([self.c_att_q(q),
                                                   self.c_att_k(k),
                                                   self.c_att_v(v) if 'mha' not in
                                                   self.input_ffn else v],
                                                  axis=2)
        if 'STE' in self.input_ffn:
            if 'Q' in self.input_ffn:
                q = (self.ste_drop_q1(self.c_att_q(x)) +
                     self.ste_drop_q2(self.c_att_q2(x)) +
                     self.ste_drop_q3(self.c_att_q3(x)) +
                     self.ste_drop_q4(self.c_att_q4(x))) / 4
            else:
                q = self.c_att_q(x)
            if 'K' in self.input_ffn:
                k = (self.ste_drop_k1(self.c_att_k(x)) +
                     self.ste_drop_k2(self.c_att_k2(x)) +
                     self.ste_drop_k3(self.c_att_k3(x)) +
                     self.ste_drop_k4(self.c_att_k4(x))) / 4
            else:
                k = self.c_att_k(x)
            if 'V' in self.input_ffn:
                v = (self.ste_drop_v1(self.c_att_v(x)) +
                     self.ste_drop_v2(self.c_att_v2(x)) +
                     self.ste_drop_v3(self.c_att_v3(x)) +
                     self.ste_drop_v4(self.c_att_v4(x))) / 4
            else:
                v = self.c_att_v(x)
            output = tf.keras.backend.concatenate([q, k, v], axis=2)

        output = self.attn(output) if mask is None else \
            self.attn([output, mask])

        if self.output_projection:
            output = self.c_attn_proj(output)

        return output


class MhaPoolLayer:
    def __init__(self, n_state: int, n_head: int, attention_dropout: float,
                 use_attn_mask: bool, layer_id: int, neg_inf: float,
                 output_projection: bool, output_dim: int,
                 input_ffn, input_ffn_dim: int, gated_ffn: bool,
                 kernel_initializer, kernel_constraint
                 ) -> None:
        self.gated_ffn = gated_ffn
        self.kernel_initializer = kernel_initializer
        self.kernel_constraint = kernel_constraint
        self.attention = MultiHeadSelfAttentionPool(n_state, n_head,
                                                    attention_dropout,
                                                    use_attn_mask, layer_id,
                                                    neg_inf, output_projection,
                                                    output_dim,
                                                    input_ffn, input_ffn_dim,
                                                    kernel_initializer,
                                                    kernel_constraint)
        if self.gated_ffn:
            self.ffn = PositionWiseFF(n_state, n_state, layer_id, True)
            self.gate = Dense(n_state, use_bias=True,
                              activation=tf.keras.backend.sigmoid,
                              kernel_initializer=self.kernel_initializer,
                              kernel_constraint=self.kernel_constraint,
                              name='layer_{}/mha_pool_gate'.format(layer_id),
                              modifications={'use_bias': False})

    def __call__(self, x, mask):
        out = self.attention(x, mask)
        if self.gated_ffn:
            y = self.ffn(x)
            out += y*self.gate(tf.keras.backend.concatenate([out, y], axis=2))
        return out


def create_mha_pool(embedding_dim: int = 768, max_len: int = 512,
                    num_heads: int = 12, num_layers: int = 12,
                    attention_dropout: float = 0.1, use_attn_mask: bool = True,
                    neg_inf: float = -1e9, internal_dim: int = 768,
                    output_projection: bool = False, output_dim: int = 768,
                    input_ffn=None, input_ffn_dim=768, gated_ffn: bool = False,
                    kernel_initializer='glorot_uniform',
                    kernel_constraint=None,
                    use_dense_connection: bool = False
                    ) -> tensorflow.keras.Model:
    input_ = Input(batch_shape=(None, max_len, embedding_dim),
                   name='input', dtype='float32')
    attn_mask = Input(batch_shape=(None, 1, max_len, max_len),
                      name='attention_mask_input', dtype=K.floatx()
                      ) if use_attn_mask else None
    inputs = [input_]
    x = input_
    for i in range(num_layers):
        out = MhaPoolLayer(internal_dim, num_heads, attention_dropout,
                           use_attn_mask, i, neg_inf, output_projection,
                           output_dim, input_ffn, input_ffn_dim, gated_ffn,
                           kernel_initializer, kernel_constraint
                           )(x, attn_mask)
        if use_dense_connection:
            x = tf.keras.backend.concatenate([x, out], axis=2)
        else:
            x = out
    if use_attn_mask:
        inputs.append(attn_mask)
    return tensorflow.keras.Model(inputs=inputs, outputs=[x], name='MhaPool')
