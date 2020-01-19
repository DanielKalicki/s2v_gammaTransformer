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
                'output_mha': False,
                'output_mha_num_heads': 1,  # used only if output_mha = True
                'output_projection': True,
            },
            'ffn_modifications': {
                'ffn_layer': True,
            }
        },
        'pooling': {
            'input': 'gt',  # concatenated input to the pooling layer
                            # gt - gated transformer (always enabled)
                            # s - sentences
            'pooling_method': 'mha',  # ['mha', 'dense']
            # 'dense': {
            #     'layer_cnt': 1,
            #     'hidden_activation': 'gelu'  # last layers has no activation
            #     'inner_dim': 4096,
            # }
            'mha': {
                'inner_dim': 4096,
                'num_layers': 1,
                'num_heads': 32,
                'attention_dropout': 0.0,
                'output_projection': False,
                'output_dim': 4096,  # used only if output_projection = True
                'input_ffn': '',  # None, q, k, v or any combination
                'input_ffn_dim': 4096,  # used only if input_ffn != None
                'gated_ffn': False,
                'use_dense_connection': False,
            },
            'pooling_activation': None,  # activation function used before pool
            'pooling_function': 'max',  # ['mean', 'max', 'l2', 'mean_max']
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
        'loss_mean0_s2v': False,
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
# -------------------------------- Config 10 ---------------------------------
configs[10]['sentence_encoder']['pooling']['mha']['num_layers'] = 2
configs[10]['sentence_encoder']['pooling']['mha']['use_dense_connection'] = \
    True
configs[10]['name'] = 'bL16_4xTr_MhaFfn_g(x,y)*y+x_' + \
    'MhaPool2LayersDense_MaxPool_4096d_Snli'
# -------------------------------- Config 11 ---------------------------------
configs[11]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'v'
configs[11]['sentence_encoder']['pooling']['pooling_activation'] = \
    tf.keras.activations.tanh
configs[11]['name'] = 'bL16_4xTr_MhaFfn_g(x,y)*y+x_' + \
    'MhaPoolVnLin_TanhB4Pool_MaxPool_4096d_Snli'
# -------------------------------- Config 12 ---------------------------------
configs[12]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'mha'
configs[12]['name'] = 'bL16_4xTr_MhaFfn_g(x,y)*y+x_' + \
    'MhaPoolVmha_MaxPool_4096d_Snli'
# -------------------------------- Config 13 ---------------------------------
configs[13]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'v'
configs[13]['sentence_encoder']['pooling']['pooling_function'] = 'l2'
configs[13]['name'] = 'bL16_4xTr_MhaFfn_g(x,y)*y+x_' + \
    'MhaPoolVnLin_L2Pool_4096d_Snli'
# -------------------------------- Config 14 ---------------------------------
configs[14]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'v'
(configs[14]['sentence_encoder']['transformer']['mha_modifications']
            ['hidden_dim']) = 4096
configs[14]['name'] = 'bL16_4xTr_MhaFfn4096h_g(x,y)*y+x_' + \
    'MhaPoolVnLin_MaxPool_4096d_Snli'
# -------------------------------- Config 15 ---------------------------------
configs[15]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'v'
(configs[15]['sentence_encoder']['transformer']['mha_modifications']
            ['hidden_layer']) = False
(configs[15]['sentence_encoder']['transformer']['mha_modifications']
            ['output_projection']) = False
configs[15]['name'] = 'bL16_4xTr_MhaNoOutProj_g(x,y)*y+x_' + \
    'MhaPoolVnLin_MaxPool_4096d_Snli'
# -------------------------------- Config 16 ---------------------------------
configs[16]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'v'
(configs[16]['sentence_encoder']['transformer']['mha_modifications']
            ['activation_after_mha']) = 'gelu'
(configs[16]['sentence_encoder']['transformer']['mha_modifications']
            ['hidden_layer']) = False
configs[16]['name'] = 'bL16_4xTr_MhaGeluOutProj_g(x,y)*y+x_' + \
    'MhaPoolVnLin_MaxPool_4096d_Snli'
# -------------------------------- Config 17 ---------------------------------
configs[17]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'v'
(configs[17]['sentence_encoder']['transformer']['mha_modifications']
            ['hidden_layer']) = False
configs[17]['name'] = 'bL16_4xTr_MhaStandard_g(x,y)*y+x_' + \
    'MhaPoolVnLin_MaxPool_4096d_Snli'
# -------------------------------- Config 18 ---------------------------------
configs[18]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'v'
configs[18]['training']['loss_mean0_s2v'] = True
configs[18]['name'] = 'bL16_4xTr_MhaFfn_g(x,y)*y+x_' + \
    'MhaPoolVnLin_MaxPool_4096d_Snli_s2vLoss'
# -------------------------------- Config 19 ---------------------------------
configs[19]['sentence_encoder']['pooling']['mha']['inner_dim'] = 2048
configs[19]['sentence_encoder']['pooling']['mha']['num_heads'] = 16
configs[19]['sentence_encoder']['pooling']['mha']['output_dim'] = 2048
configs[19]['sentence_encoder']['pooling']['mha']['input_ffn_dim'] = 2048
configs[19]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'v'
configs[13]['sentence_encoder']['pooling']['pooling_function'] = 'mean_max'
configs[19]['name'] = 'bL16_4xTr_MhaFfn_g(x,y)*y+x_' + \
    'MhaPoolVnLin_MeanMaxPool_4096d_Snli'
# -------------------------------- Config 20 ---------------------------------
configs[20]['sentence_encoder']['pooling']['mha']['gated_ffn'] = True
configs[20]['name'] = 'bL16_4xTr_MhaFfn_g(x,y)*y+x_' + \
    'MhaPoolGatedFfn_MaxPool_4096d_Snli'
# -------------------------------- Config 21 ---------------------------------
configs[21]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'v'
(configs[21]['sentence_encoder']['transformer']['mha_modifications']
            ['hidden_layer']) = False
(configs[21]['sentence_encoder']['transformer']['mha_modifications']
            ['output_mha']) = True
(configs[21]['sentence_encoder']['transformer']['mha_modifications']
            ['output_projection']) = False
configs[21]['name'] = 'bL16_4xTr_Mha16hMha1h_g(x,y)*y+x_' + \
    'MhaPoolVnLin_MaxPool_4096d_Snli'
# -------------------------------- Config 22 ---------------------------------
configs[22]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'v'
(configs[22]['sentence_encoder']['transformer']['mha_modifications']
            ['hidden_layer']) = False
(configs[22]['sentence_encoder']['transformer']['mha_modifications']
            ['output_mha']) = True
(configs[22]['sentence_encoder']['transformer']['mha_modifications']
            ['output_mha_num_heads']) = 16
(configs[22]['sentence_encoder']['transformer']['mha_modifications']
            ['output_projection']) = False
configs[22]['name'] = 'bL16_4xTr_Mha16hMha16h_g(x,y)*y+x_' + \
    'MhaPoolVnLin_MaxPool_4096d_Snli'
# -------------------------------- Config 23 ---------------------------------
configs[23]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'v'
(configs[23]['sentence_encoder']['transformer']['mha_modifications']
            ['hidden_layer']) = False
(configs[23]['sentence_encoder']['transformer']['mha_modifications']
            ['output_mha']) = True
(configs[23]['sentence_encoder']['transformer']['mha_modifications']
            ['output_mha_num_heads']) = 16
(configs[23]['sentence_encoder']['transformer']['mha_modifications']
            ['output_projection']) = True
configs[23]['name'] = 'bL16_4xTr_Mha16hMha16hOutProj_g(x,y)*y+x_' + \
    'MhaPoolVnLin_MaxPool_4096d_Snli'
# -------------------------------- Config 24 ---------------------------------
configs[24]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'v'
configs[24]['sentence_encoder']['pooling']['mha']['num_heads'] = 64
configs[24]['name'] = 'bL16_4xTr_MhaFfn_g(x,y)*y+x_' + \
    'MhaPoolVnLin64h_MaxPool_4096d_Snli'
# -------------------------------- Config 25 ---------------------------------
configs[25]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'v'
configs[25]['sentence_encoder']['pooling']['mha']['num_heads'] = 16
configs[25]['name'] = 'bL16_4xTr_MhaFfn_g(x,y)*y+x_' + \
    'MhaPoolVnLin16h_MaxPool_4096d_Snli'
# -------------------------------- Config 26 ---------------------------------
# hardcoded - output_mha is parallel
configs[26]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'v'
(configs[26]['sentence_encoder']['transformer']['mha_modifications']
            ['output_mha']) = True
(configs[26]['sentence_encoder']['transformer']['mha_modifications']
            ['output_mha_num_heads']) = 1
configs[26]['name'] = 'bL16_4xTr_Mha16hFfnParMha1h_g(x,y)*y+x_' + \
    'MhaPoolVnLin_MaxPool_4096d_Snli'
# -------------------------------- Config 27 ---------------------------------
# hardcoded
configs[27]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'v'
configs[27]['sentence_encoder']['pooling']['mha']['gated_ffn'] = True
configs[27]['name'] = 'bL16_4xTr_MhaFfn_g(x,y)*y+x_' + \
    'MhaPoolVnLinResidualFfn_MaxPool_4096d_Snli'
# -------------------------------- Config 28 ---------------------------------
configs[28]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'v'
configs[28]['sentence_encoder']['pooling']['input'] = 'gt,s'
configs[28]['name'] = 'bL16_4xTr_MhaFfn_g(x,y)*y+x_' + \
    'gtsInput_MhaPoolVnLin_MaxPool_4096d_Snli'
# -------------------------------- Config 29 ---------------------------------
configs[29]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'v'
(configs[29]['sentence_encoder']['transformer']['mha_modifications']
            ['output_mha']) = True
(configs[29]['sentence_encoder']['transformer']['mha_modifications']
            ['output_mha_num_heads']) = 16
configs[29]['name'] = 'bL16_4xTr_Mha16hMha16hFfn_g(x,y)*y+x_' + \
    '_MhaPoolVnLin_MaxPool_4096d_Snli'
# -------------------------------- Config 30 ---------------------------------
configs[30]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'v'
configs[30]['sentence_encoder']['transformer']['num_layers'] = 6
(configs[30]['sentence_encoder']['transformer']['ffn_modifications']
            ['ffn_layer']) = False
configs[30]['name'] = 'bL16_6xTr_Mha16hFfn_g(x,y)*y+x_nFfnLayer' + \
    '_MhaPoolVnLin_MaxPool_4096d_Snli'
# -------------------------------- Config 31 ---------------------------------
configs[31]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'v'
configs[31]['sentence_encoder']['transformer']['num_layers'] = 8
(configs[31]['sentence_encoder']['transformer']['ffn_modifications']
            ['ffn_layer']) = False
configs[31]['name'] = 'bL16_8xTr_Mha16hFfn_g(x,y)*y+x_nFfnLayer' + \
    '_MhaPoolVnLin_MaxPool_4096d_Snli'
# gate Mha1 in MhaPool
