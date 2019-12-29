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


class MultiHeadSelfAttention:
    """
    Multi Head Self Attention implementation used in Transformer architecture.

    Architecture:
          +-------+      +-------+
          |       +----->+       |     +----------+     +-------+
          |       |    q |       |     |          |     |       |
          |       |      |       |     |Activation|     |       |
    +---->+ Dense +----->+  MHA  +---->+function  +---->+ Dense +---->
     x    |       |    k |       |     |    *     |     |   *   |   y
          |       |      |       |     |          |     |       |
          |       +----->+       |     +----------+     +-------+
          +-------+    v +-------+
    * - optional
    """
    def __init__(self, n_state: int, n_head: int, attention_dropout: float,
                 use_attn_mask: bool, layer_id: int, neg_inf: float,
                 modifications={}) -> None:
        assert n_state % n_head == 0
        self.n_state = n_state
        self.act_after_mha = False
        self.out_proj = False

        self.c_attn = Dense(3 * n_state, use_bias=False,
                            name='layer_{}/c_attn'.format(layer_id))

        self.attn = MultiHeadAttention(n_head, n_state, attention_dropout,
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

        if ('output_projection' in modifications) \
           and (modifications['output_projection']):
            self.out_proj = True
            self.c_attn_proj = Dense(n_state, use_bias=False,
                                     name='layer_{}/c_attn_proj'
                                          .format(layer_id))

    def __call__(self, x, mask):
        output = self.c_attn(x)
        output = self.attn(output) if mask is None else \
            self.attn([output, mask])
        if self.act_after_mha:
            output = self.activation(output)
        if self.out_proj:
            output = self.c_attn_proj(output)
        return output


class PositionWiseFF:
    """
    Feedforward network implementation used in Transformer architecture.
    """
    def __init__(self, n_state: int, d_hid: int, layer_id: int,
                 accurate_gelu: bool) -> None:
        self.c_fc = Dense(d_hid, use_bias=False,
                          name='layer_{}/c_fc'.format(layer_id))
        self.activation = Gelu(accurate=accurate_gelu,
                               name='layer_{}/gelu'.format(layer_id))
        self.c_ffn_proj = Dense(n_state, use_bias=False,
                                name='layer_{}/c_ffn_proj'.format(layer_id))

    def __call__(self, x):
        output = self.c_fc(x)
        output = self.activation(output)
        output = self.c_ffn_proj(output)
        return output


class GatedEncoderLayer:
    def __init__(self, n_state: int, n_head: int, d_hid: int,
                 residual_dropout: float, attention_dropout: float,
                 use_attn_mask: bool, layer_id: int, neg_inf: float,
                 ln_epsilon: float, accurate_gelu: bool,
                 gate_type='None', mha_modifications={}) -> None:
        self.gate_type = gate_type

        self.attention = MultiHeadSelfAttention(n_state, n_head,
                                                attention_dropout,
                                                use_attn_mask, layer_id,
                                                neg_inf, mha_modifications)
        self.drop1 = Dropout(residual_dropout,
                             name='layer_{}/ln_1_drop'.format(layer_id))
        self.ln1 = LayerNormalization(ln_epsilon,
                                      name='layer_{}/ln_1'.format(layer_id))
        self.ffn = PositionWiseFF(n_state, d_hid, layer_id, accurate_gelu)
        self.drop2 = Dropout(residual_dropout,
                             name='layer_{}/ln_2_drop'.format(layer_id))
        self.ln2 = LayerNormalization(ln_epsilon,
                                      name='layer_{}/ln_2'.format(layer_id))

        if self.gate_type != 'None':
            self.gate1_dense = Dense(n_state, use_bias=True,
                                     activation=tf.keras.backend.sigmoid,
                                     name='layer_{}/gate1'.format(layer_id))
            self.gate2_dense = Dense(n_state, use_bias=True,
                                     activation=tf.keras.backend.sigmoid,
                                     name='layer_{}/gate2'.format(layer_id))
            if self.gate_type == 'Wg(y)*tanh(Ug(y)) + x':
                self.gate1_Ug = Dense(n_state, use_bias=False,
                                      activation=tf.keras.backend.tanh,
                                      name='layer_{}/gate1_Ug'
                                           .format(layer_id))
                self.gate2_Ug = Dense(n_state, use_bias=False,
                                      activation=tf.keras.backend.tanh,
                                      name='layer_{}/gate2_Ug'
                                           .format(layer_id))

    def gate_output(self, x, y, g_dense, drop):
        if self.gate_type == 'Wg(x)*y + x':
            # g(x, y) = σ(Wg*x + b)*y + x
            g_var = x
        elif self.gate_type == 'Wg(x,y)*y + x':
            # g(x, y) = σ(Wg*(x||y) + b)*y + x
            g_var = tf.keras.backend.concatenate([x, y], axis=2)
        return drop(g_dense(g_var) * y) + x

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
        #          +-----+  y
        #    +---->+ MHA +------+
        #    |     +-----+      |
        #    |                  v
        #    |               +--+---+     +----+
        # +--+-------------->+ Gate +---->+ LN +---->
        # x                  +------+     +----+    x
        y = self.attention(x, mask)

        if self.gate_type in ['Wg(x)*y + x', 'Wg(x,y)*y + x']:
            x = self.gate_output(x, y, self.gate1_dense, self.drop1)
        elif self.gate_type == 'Wg(y)*tanh(Ug(y)) + x':
            x = self.gate_sigmoid_tanh(x, y, self.gate1_dense, self.gate1_Ug,
                                       self.drop1)
        elif self.gate_type == 'Wg(x)*x + y':
            x = self.gate_input(x, y, self.gate1_dense, self.drop1)
        elif self.gate_type == 'Wg(x)*x + (1-Wg(x))*y':
            x = self.gate_highway(x, y, self.gate1_dense, self.drop1)
        elif self.gate_type == 'None':
            x = x + self.drop1(y)

        x = self.ln1(x)

        #          +-----+  y
        #    +---->+ FFN +------+
        #    |     +-----+      |
        #    |                  v
        #    |               +--+---+     +----+
        # +--+-------------->+ Gate +---->+ LN +---->
        # x                  +------+     +----+    x
        y = self.ffn(x)

        if self.gate_type in ['Wg(x)*y + x', 'Wg(x,y)*y + x']:
            x = self.gate_output(x, y, self.gate2_dense, self.drop2)
        elif self.gate_type == 'Wg(y)*tanh(Ug(y)) + x':
            x = self.gate_sigmoid_tanh(x, y, self.gate2_dense, self.gate2_Ug,
                                       self.drop2)
        elif self.gate_type == 'Wg(x)*x + y':
            x = self.gate_input(x, y, self.gate2_dense, self.drop2)
        elif self.gate_type == 'Wg(x)*x + (1-Wg(x))*y':
            x = self.gate_highway(x, y, self.gate2_dense, self.drop2)
        elif self.gate_type == 'None':
            x = x + self.drop2(y)

        x = self.ln2(x)

        return x


def create_gated_transformer(embedding_dim: int = 768, max_len: int = 512,
                             num_heads: int = 12, num_layers: int = 12,
                             attention_dropout: float = 0.0,
                             d_hid: int = 768 * 4,
                             residual_dropout: float = 0.0,
                             use_attn_mask: bool = True, neg_inf: float = -1e9,
                             layer_norm_epsilon: float = 1e-5,
                             accurate_gelu: bool = True, gate_type='None',
                             mha_modifications={}) -> tensorflow.keras.Model:

    if gate_type not in ['None', 'Wg(x)*y + x', 'Wg(x,y)*y + x',
                         'Wg(y)*tanh(Ug(y)) + x', 'Wg(x)*x + y',
                         'Wg(x)*x + (1-Wg(x))*y']:
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
                              accurate_gelu, gate_type, mha_modifications
                              )(x, attn_mask)
    if use_attn_mask:
        inputs.append(attn_mask)
    return tensorflow.keras.Model(inputs=inputs, outputs=[x],
                                  name='GatedTransformer')


class MultiHeadSelfAttentionPool:
    """
    Modified Multi Head Attention used to expand the word embeddings before
        pooling.

    Architecture:
         +-------+                               +-------+
         |       +------------------------------>+       |   +------------+
         |       |                             q |       |   |            |
         |       |                               |       |   | Output     |
    +--->+ Dense +------------------------------>+  MHA  +-->+ Projection +--->
      x  |       |   +----------+   +-------+  k |       |   |     *      |  y
         |       |   |          |   |       |    |       |   |            |
         |       +-->+Activation+-->+ Dense +--->+       |   +------------+
         +-------+   |function  |   |   *   |  v +-------+
                     |    *     |   +-------+
                     +----------+
     * - optional

    """
    def __init__(self, n_state: int, n_head: int, attention_dropout: float,
                 use_attn_mask: bool, layer_id: int, neg_inf: float,
                 output_projection: bool, output_dim: int) -> None:
        assert n_state % n_head == 0
        self.n_state = n_state
        self.output_projection = output_projection

        self.c_att_v = Dense(n_state, use_bias=False, activation=None,
                             name='layer_{}/c_att_v'.format(layer_id))
        self.c_att_v2 = Dense(n_state, use_bias=False, activation=gelu,
                              name='layer_{}/c_att_v2'.format(layer_id))
        self.c_att_k = Dense(n_state, use_bias=False, activation=None,
                             name='layer_{}/c_att_k'.format(layer_id))
        self.c_att_q = Dense(n_state, use_bias=False, activation=None,
                             name='layer_{}/c_att_q'.format(layer_id))

        self.attn = MultiHeadAttention(n_head, n_state, attention_dropout,
                                       use_attn_mask, neg_inf,
                                       name='layer_{}/self_attention'
                                            .format(layer_id))
        if self.output_projection:
            self.c_attn_proj = Dense(output_dim, use_bias=False,
                                     name='layer_{}/c_attn_proj'
                                          .format(layer_id))

    def __call__(self, x, mask):
        output = tf.keras.backend.concatenate([self.c_att_q(x),
                                               self.c_att_k(x),
                                               self.c_att_v(self.c_att_v2(x))],
                                              axis=2)

        output = self.attn(output) if mask is None else \
            self.attn([output, mask])

        if self.output_projection:
            output = self.c_attn_proj(output)

        return output


class MhaPoolLayer:
    def __init__(self, n_state: int, n_head: int, attention_dropout: float,
                 use_attn_mask: bool, layer_id: int, neg_inf: float,
                 output_projection: bool, output_dim: int) -> None:
        self.attention = MultiHeadSelfAttentionPool(n_state, n_head,
                                                    attention_dropout,
                                                    use_attn_mask, layer_id,
                                                    neg_inf, output_projection,
                                                    output_dim)

    def __call__(self, x, mask):
        return self.attention(x, mask)


def create_mha_pool(embedding_dim: int = 768, max_len: int = 512,
                    num_heads: int = 12, num_layers: int = 12,
                    attention_dropout: float = 0.1, use_attn_mask: bool = True,
                    neg_inf: float = -1e9, internal_dim: int = 768,
                    output_projection: bool = False, output_dim: int = 768
                    ) -> tensorflow.keras.Model:
    input_ = Input(batch_shape=(None, max_len, embedding_dim),
                   name='input', dtype='float32')
    attn_mask = Input(batch_shape=(None, 1, max_len, max_len),
                      name='attention_mask_input', dtype=K.floatx()
                      ) if use_attn_mask else None
    inputs = [input_]
    x = input_
    for i in range(num_layers):
        x = MhaPoolLayer(internal_dim, num_heads, attention_dropout,
                         use_attn_mask, i, neg_inf, output_projection,
                         output_dim)(x, attn_mask)
    if use_attn_mask:
        inputs.append(attn_mask)
    return tensorflow.keras.Model(inputs=inputs, outputs=[x], name='MhaPool')
