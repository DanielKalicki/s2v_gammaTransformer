import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense
from nlp_blocks.transformer.transformer import (create_mha_pool,
                                                create_gated_transformer)
from nlp_blocks.transformer.funcs import gelu


class SentenceEncoderModel(tf.keras.Model):
    """
    Keras model of Sentence Encoder.

    Class used to encode input sentences into fix length vectors.
    Ex. I am Groot -> [0.324, -0.153, 0.988,  ... , 0.002]
    """
    def __init__(self, config):
        super(SentenceEncoderModel, self).__init__()
        self.config = config

        max_sent_len = self.config['max_sent_len']
        word_dim = self.config['sentence_encoder']['transformer']['word_dim']

        self.pooling_method = (self.config['sentence_encoder']['pooling']
                                          ['pooling_method'])
        self.pooling_activation = (self.config['sentence_encoder']['pooling']
                                              ['pooling_activation'])
        if self.pooling_activation == 'gelu':
            self.pooling_activation = gelu
        self.pooling_function = (self.config['sentence_encoder']['pooling']
                                            ['pooling_function'])
        self.pool_projection = (self.config['sentence_encoder']
                                           ['pool_projection'])

        self.s2v_dim = self.config['s2v_dim']

        self.input_drop = Dropout(self.config['sentence_encoder']
                                             ['input_drop'])

        gtr_params = {
            'max_len': max_sent_len,
            'num_layers': (self.config['sentence_encoder']['transformer']
                                      ['num_layers']),
            'num_heads': (self.config['sentence_encoder']['transformer']
                                     ['num_heads']),
            'attention_dropout': (self.config['sentence_encoder']
                                             ['transformer']
                                             ['attention_dropout']),
            'residual_dropout': (self.config['sentence_encoder']['transformer']
                                            ['residual_dropout']),
            'embedding_dim': word_dim,
            'd_hid': (self.config['sentence_encoder']['transformer']
                                 ['ffn_dim']),
            'use_attn_mask': True,
            'gate_type': (self.config['sentence_encoder']['transformer']
                                     ['gate_type']),
            'mha_modifications': (self.config['sentence_encoder']
                                             ['transformer']
                                             ['mha_modifications']),
        }
        self.gtr = create_gated_transformer(**gtr_params)

        if self.pooling_method == 'mha':
            mha_params = {
                'max_len': max_sent_len,
                'num_layers': 1,
                'num_heads': (self.config['sentence_encoder']['pooling']
                                         ['mha']['num_heads']),
                'attention_dropout': (self.config['sentence_encoder']
                                                 ['pooling']
                                                 ['mha']['attention_dropout']),
                'embedding_dim': word_dim,
                'use_attn_mask': True,
                'internal_dim': (self.config['sentence_encoder']['pooling']
                                            ['mha']['inner_dim']),
                'output_projection': (self.config['sentence_encoder']
                                                 ['pooling']
                                                 ['mha']['output_projection']),
                'output_dim': (self.config['sentence_encoder']['pooling']
                                          ['mha']['output_dim']),
                'input_ffn': (self.config['sentence_encoder']['pooling']
                                         ['mha']['input_ffn']),
                'input_ffn_dim': (self.config['sentence_encoder']['pooling']
                                             ['mha']['input_ffn_dim']),
            }
            self.mha_pool = create_mha_pool(**mha_params)

        elif self.pooling_method == 'dense':
            self.dense_pool_layer_cnt = (self.config['sentence_encoder']
                                                    ['pooling']
                                                    ['dense']['layer_cnt'])
            self.dense_pool_act = (self.config['sentence_encoder']['pooling']
                                              ['dense']['hidden_activation'])
            self.dense_pool_dim = (self.config['sentence_encoder']['pooling']
                                              ['dense']['inner_dim'])
            self.dense_pool_layers = []
            for i in range(self.dense_pool_layer_cnt):
                dense_act = (self.dense_pool_act
                             if i != (self.dense_pool_layer_cnt-1)
                             else None)
                self.dense_pool_layers.append(Dense(self.dense_pool_dim,
                                                    use_bias=False,
                                                    activation=dense_act,
                                                    name='dense_pool_{}'
                                                         .format(i)))
        if self.pool_projection:
            self.s2v_projection = Dense(self.s2v_dim, activation=None,
                                        use_bias=False)

    def call(self, sentence, sentence_mask, sentence_transformer_mask):
        """
        Encode input sentence into fix lenght vector.

        @param self: The Object pointer.
        @type sentence: Tensor (batch size, sentence length,
                                word embeddings dimension).
        @param sentence: Input sentence to be embedded into sentence vector.
        @type sentence_mask: Boolean tensor (batch size, sentence length)
        @param sentence_mask: Mask of words embeddings.
        @type sentence_transformer_mask: Boolean tensor
            (batch size, 1, sentence length, sentence_length).
        @param sentence_transformer_mask: Attention mask applied in
            Transformer's MHA layer.

        @rtype: Tensor (batch size, sentence vector dimension).
        @return: Sentence vector generated from input.
        """
        sent = self.input_drop(sentence)
        sent_mask = tf.expand_dims(sentence_mask, axis=-1)
        sent_tr_mask = sentence_transformer_mask

        # ---------------------------------------------------------------------
        # Gated transformer
        # ---------------------------------------------------------------------
        sent_out = self.gtr([sent, sent_tr_mask])

        # ---------------------------------------------------------------------
        # Pooling preparation - expanding and mixing
        # ---------------------------------------------------------------------
        if self.pooling_method == 'mha':
            sent_pool = self.mha_pool([sent_out, sent_tr_mask])
        elif self.pooling_method == 'dense':
            sent_pool = sent_out
            for i in range(self.dense_pool_layer_cnt):
                sent_pool = self.dense_pool_layers[i](sent_pool)

        # ---------------------------------------------------------------------
        # Activation function before Pool
        # ---------------------------------------------------------------------
        if self.pooling_activation:
            sent_pool = self.pooling_activation(sent_pool)

        # ---------------------------------------------------------------------
        # Pool function - mean or max
        # ---------------------------------------------------------------------
        if self.pooling_function == 'mean':
            s2v_pool = tf.reduce_sum(sent_pool*sent_mask, axis=1) \
                       / tf.reduce_sum(sent_mask, axis=1)
        elif self.pooling_function == 'max':
            s2v_pool = tf.reduce_max(sent_pool + (1-sent_mask)*-1e5, axis=1)

        # ---------------------------------------------------------------------
        # Pool projection to s2v dimension
        # ---------------------------------------------------------------------
        if self.pool_projection:
            s2v = self.s2v_projection(s2v_pool)
        else:
            s2v = s2v_pool

        print("s2v:\t"+str(s2v))
        return s2v
