import tensorflow as tf
import copy

default_config = {
    'batch_size': 16,
    'max_sent_len': 64,
    'word_edim': 1024,
    's2v_dim': 4096,

    'sentence_encoder': {
        'input_drop': 0.0,
        'transformer': {
            'word_dim': 1024,
            'num_layers': 4,
            'num_heads': 16,
            'ffn_dim': 4*1024,
            'residual_dropout': 0.0,
            'attention_dropout': 0.0,
            'gate_type': 'Wg(x,y)*y + x',  # 'None' - Residual connections
                                           # 'Wg(x)*y + x'
                                           # 'Wg(x,y)*y + x'
                                           # 'Wg(y)*tanh(Ug(y)) + x'
                                           # 'Wg(x)*x + y'
                                           # 'Wg(x)*x + (1-Wg(x))*y'
            'mha_modifications': {
                'inner_dim': 1024,
                'activation_after_mha': None,
                'hidden_layer': True,
                'hidden_dim': 1024,  # used only if hidden_layer = True
                'hidden_activation': 'gelu',  # used only if hidden_layer = True
                'output_projection': True
            }
        },
        'pooling': {
            'pooling_method': 'mha',  # ['mha', 'dense']
            # 'dense': {
            #     'layer_cnt': 1,
            #     'hidden_activation': 'gelu'  # last layers has no activation
            #     'inner_dim': 4096,
            # }
            'mha': {
                'inner_dim': 4096,
                'num_heads': 32,
                'attention_dropout': 0.0,
                'output_projection': False,
                'output_dim': 4096,  # used only if output_projection = True
                'input_ffn': None,  # None, q, k, v or any combination
                'input_ffn_dim': 4096  # used only if input_ffn != None
            },
            'pooling_activation': None,  # activation function used before pool
            'pooling_function': 'max',  # ['mean', 'max']
        },
        'pool_projection': False,  # if True pooling output will be projected
                                   #    to s2v_dim
    },

    'classifier_network': {
        'hidden_dim': 512,
        'hidden_layer_cnt': 1,
        'dropout': 0.0,
        'hidden_layer_norm': False,
        'hidden_activation': 'gelu',
        'prediction_activation': tf.keras.activations.softmax,
    },

    'training': {
        'optimizer': 'Nadam',
        'clipnorm': 1.,
        'lr': ([4e-5]*5 + [2e-5]*6 + [1e-5]*5 + [5e-6]*6 + [2e-6]*3 + [1e-6]*3
               + [5e-7]*2 + [2e-7]*2 + [1e-7]*6),
        'label_smoothing': 0.2,
        'label_noise': 0.0
    }
}

configs = []
for i in range(100):
    configs.append(copy.deepcopy(default_config))

# -------------------------------- Config 00 ---------------------------------
configs[0]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'v'
configs[0]['classifier_network']['hidden_activation'] = \
    tf.keras.activations.tanh
configs[0]['name'] = 'bL16_4xTr_MhaFfn_g(x,y)*y+x_' + \
    'MhaPoolVnLin_MaxPool_4096d_Snli_TanhClass'
# -------------------------------- Config 01 ---------------------------------
# hardcoded code change
configs[1]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'v'
configs[1]['name'] = 'bL16_4xTr_MhaFfn_g(x,y)*y+x_' + \
    'MhaPoolVnLin_MaxPool_4096d_Snli_TrBias'
# -------------------------------- Config 02 ---------------------------------
configs[2]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'v'
configs[2]['classifier_network']['dropout'] = 0.1
configs[2]['name'] = 'bL16_4xTr_MhaFfn_g(x,y)*y+x_' + \
    'MhaPoolVnLin_MaxPool_4096d_Snli_ClassDrop.1'
# -------------------------------- Config 03 ---------------------------------
configs[3]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'v'
configs[3]['classifier_network']['hidden_layer_norm'] = True
configs[3]['name'] = 'bL16_4xTr_MhaFfn_g(x,y)*y+x_' + \
    'MhaPoolVnLin_MaxPool_4096d_Snli_LnClass'
# -------------------------------- Config 04 ---------------------------------
configs[4]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'v'
configs[4]['classifier_network']['hidden_activation'] = \
    tf.keras.activations.relu
configs[4]['name'] = 'bL16_4xTr_MhaFfn_g(x,y)*y+x_' + \
    'MhaPoolVnLin_MaxPool_4096d_Snli_ReluClass'
# -------------------------------- Config 05 ---------------------------------
configs[5]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'v'
configs[5]['sentence_encoder']['transformer']['num_layers'] = 3
configs[5]['name'] = 'bL16_3xTr_MhaFfn_g(x,y)*y+x_' + \
    'MhaPoolVnLin_MaxPool_4096d_Snli'
# -------------------------------- Config 06 ---------------------------------
configs[6]['sentence_encoder']['pooling']['mha']['input_ffn'] = ''
configs[6]['name'] = 'bL16_4xTr_MhaFfn_g(x,y)*y+x_' + \
    'MhaPool_MaxPool_4096d_Snli'
# -------------------------------- Config 07 ---------------------------------
configs[7]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'q'
configs[7]['name'] = 'bL16_4xTr_MhaFfn_g(x,y)*y+x_' + \
    'MhaPoolQnLin_MaxPool_4096d_Snli'
# -------------------------------- Config 08 ---------------------------------
configs[8]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'k'
configs[8]['name'] = 'bL16_4xTr_MhaFfn_g(x,y)*y+x_' + \
    'MhaPoolKnLin_MaxPool_4096d_Snli'
# -------------------------------- Config 09 ---------------------------------
configs[9]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'v'
configs[9]['sentence_encoder']['pooling']['mha']['input_ffn_dim'] = 4096*2
configs[9]['name'] = 'bL16_4xTr_MhaFfn_g(x,y)*y+x_' + \
    'MhaPoolVnLin8k_MaxPool_4096d_Snli'
