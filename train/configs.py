import tensorflow as tf
import copy

default_config = {
    'batch_size': 16,
    'max_sent_len': 64,
    'word_edim': 1024,
    's2v_dim': 4096,
    'name': '',

    'sentence_encoder': {
        'input_drop': 0.0,
        'transformer': {
            'word_dim': 1024,
            'num_layers': 4,
            'num_heads': 16,
            'ffn_dim': 4*1024,
            'residual_dropout': 0.0,
            'attention_dropout': 0.0,
            'kernel_initializer': tf.keras.initializers.glorot_uniform,
            'kernel_constraint': None,
            'normalization_position': 'post',  # pre, preMod or post normalization
                                               # https://openreview.net/pdf?id=B1x8anVFPr
            'gate_type': 'Wg(x,y)*y + x',  # 'None' - Residual connections
                                           # 'Wg(x)*y + x'
                                           # 'Wg(x,y)*y + x'
                                           # 'Wg(y)*tanh(Ug(y)) + x'
                                           # 'Wg(x)*x + y'
                                           # 'Wg(x)*x + (1-Wg(x))*y'
                                           # 'Ffn(x,y)*y + x'
                                           # 'FfnNac(x,y)*y + x'
                                           # 'Mha(x,y)*y + x'
                                           # 'STE(x,y)*y + x'
            'mha_modifications': {
                'use_bias': False,
                'kernel_initializer': tf.keras.initializers.glorot_uniform,
                'kernel_constraint': None,
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
                'use_bias': False,
                'ffn_layer': True,
                'gated': False,
                'small_ffn': False,
                'nac': False
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
                                  # STE Q, K, V or any combination
                'input_ffn_dim': 4096,  # used only if input_ffn != None
                'gated_ffn': False,
                'input_concat': False,
                'kernel_initializer': tf.keras.initializers.glorot_uniform,
                'kernel_constraint': None,
                'use_dense_connection': False,
            },
            'pooling_activation': None,  # activation function used before pool
            'pooling_in_dropout': 0.0,
            'pooling_out_dropout': 0.0,
            'pooling_function': 'max',  # ['mean', 'max', 'l2', 'mean_max']
        },
        'pool_projection': False,  # if True pooling output will be projected
                                   #    to s2v_dim
    },

    'classifier_network': {
        'hidden_dim': 512,
        'hidden_layer_cnt': 1,
        'dropout': 0.0,
        'hidden_nac': False,
        'hidden_layer_norm': False,
        'hidden_activation': 'gelu',
        'prediction_activation': tf.keras.activations.softmax,
        'kernel_initializer': tf.keras.initializers.glorot_uniform,
        'kernel_constraint': None,
        'gated': False,
        'shortcut': False,
        'num_classes': 3
    },

    'training': {
        'task': 'snli',
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

# ------------------------------- Config 00-03 --------------------------------
for i in range(0, 4):
    configs[i]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'v'
    configs[i]['name'] = 'bL16_4xTr_MhaFfn_g(x,y)*y+x_' + \
        'MhaPoolVnLin_MaxPool_4096d_Snli_'+str(i)
# ------------------------------- Config 04-07 --------------------------------
for i in range(4, 8):
    configs[i]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'v'
    (configs[i]['sentence_encoder']['transformer']['mha_modifications']
                ['hidden_layer']) = False
    (configs[i]['sentence_encoder']['transformer']['mha_modifications']
                ['output_projection']) = True
    configs[i]['name'] = 'bL16_4xTr_MhaOutProj_g(x,y)*y+x_' + \
        'MhaPoolVnLin_MaxPool_4096d_Snli_'+str(i)
# ------------------------------- Config 08-11 --------------------------------
for i in range(8, 12):
    configs[i]['sentence_encoder']['pooling']['mha']['input_ffn'] = ''
    configs[i]['name'] = 'bL16_4xTr_MhaFfn_g(x,y)*y+x_' + \
        'MhaPool_MaxPool_4096d_Snli_'+str(i)
# ------------------------------- Config 12-15 --------------------------------
for i in range(12, 16):
    configs[i]['sentence_encoder']['pooling']['mha']['input_ffn'] = ''
    (configs[i]['sentence_encoder']['pooling']['mha']
            ['output_projection']) = True
    configs[i]['name'] = 'bL16_4xTr_MhaFfn_g(x,y)*y+x_' + \
        'MhaPoolOutProj_MaxPool_4096d_Snli_'+str(i)
# ------------------------------- Config 16-19 --------------------------------
for i in range(16, 20):
    configs[i]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'mha'
    configs[i]['name'] = 'bL16_4xTr_MhaFfn_g(x,y)*y+x_' + \
        'MhaPoolVmha_MaxPool_4096d_Snli_'+str(i)
# ------------------------------- Config 20-23 --------------------------------
for i in range(20, 24):
    configs[i]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'v'
    (configs[i]['sentence_encoder']['transformer']['mha_modifications']
                ['output_mha']) = True
    (configs[i]['sentence_encoder']['transformer']['mha_modifications']
                ['output_mha_num_heads']) = 16
    configs[i]['name'] = 'bL16_4xTr_MhaMha16hFfn_g(x,y)*y+x_' + \
        'MhaPoolVnLin_MaxPool_4096d_Snli_'+str(i)
# ------------------------------- Config 24-27 --------------------------------
for i in range(24, 28):
    configs[i]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'v'
    configs[i]['sentence_encoder']['transformer']['gate_type'] = \
        'Ffn(x,y)*y + x'
    configs[i]['name'] = 'bL16_4xTr_MhaFfn_Ffn(x,y)*y+x_' + \
        'MhaPoolVnLin_MaxPool_4096d_Snli_'+str(i)
# ------------------------------- Config 28-31 --------------------------------
for i in range(28, 30):
    configs[i]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'mha'
    configs[i]['sentence_encoder']['transformer']['gate_type'] = \
        'Ffn(x,y)*y + x'
    configs[i]['name'] = 'bL16_4xTr_MhaFfn_Ffn(x,y)*y+x_' + \
        'MhaPoolVmha_MaxPool_4096d_Snli_'+str(i)
# -----------------------------------------------------------------------------
for i in range(31, 100):
    configs[i]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'v'
    configs[i]['sentence_encoder']['transformer']['gate_type'] = \
        'Ffn(x,y)*y + x'
    configs[i]['name'] = 'bL16_4xTr_MhaFfn_Ffn(x,y)*y+x_' + \
        'MhaPoolVnLin_MaxPool_4096d_Snli_valsnli_'+str(i)
# -----------------------------------------------------------------------------
for i in range(40, 41):
    configs[i]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'v'
    configs[i]['sentence_encoder']['pooling']['mha']['inner_dim'] = 3072
    configs[i]['sentence_encoder']['pooling']['mha']['num_heads'] = 24
    configs[i]['sentence_encoder']['pooling']['mha']['output_dim'] = 3072
    configs[i]['sentence_encoder']['pooling']['mha']['input_ffn_dim'] = 3072
    configs[i]['sentence_encoder']['pooling']['mha']['use_dense_connection'] = True
    configs[i]['sentence_encoder']['transformer']['gate_type'] = \
        'Ffn(x,y)*y + x'
    configs[i]['name'] = 'bL16_4xTr_MhaFfn_Ffn(x,y)*y+x_' + \
        'MhaPoolVnLin_MhaDenseOut_MaxPool_4096d_Snli_'+str(i)
# -----------------------------------------------------------------------------
for i in range(42, 43):
    configs[i]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'NACv'
    configs[i]['sentence_encoder']['transformer']['gate_type'] = \
        'Ffn(x,y)*y + x'
    configs[i]['name'] = 'bL16_4xTr_MhaFfn_Ffn(x,y)*y+x_' + \
        'MhaPoolVnLinNAC_MaxPool_4096d_Snli_'+str(i)
# -----------------------------------------------------------------------------
for i in range(43, 44):
    configs[i]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'v'
    configs[i]['sentence_encoder']['transformer']['gate_type'] = \
        'Ffn(x,y)*y + x'
    (configs[i]['sentence_encoder']['transformer']['ffn_modifications']
                ['nac']) = True
    configs[i]['name'] = 'bL16_4xTr_MhaFfn_FfnNac_Ffn(x,y)*y+x_' + \
        'MhaPoolVnLin_MaxPool_4096d_Snli_'+str(i)
# -----------------------------------------------------------------------------
for i in range(44, 45):
    configs[i]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'v'
    configs[i]['sentence_encoder']['transformer']['gate_type'] = \
        'FfnNac(x,y)*y + x'
    configs[i]['name'] = 'bL16_4xTr_MhaFfn_FfnNac(x,y)*y+x_' + \
        'MhaPoolVnLin_MaxPool_4096d_Snli_'+str(i)
# -----------------------------------------------------------------------------
for i in range(45, 50):
    configs[i]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'v'
    configs[i]['sentence_encoder']['transformer']['gate_type'] = \
        'Ffn(x,y)*y + x'
    configs[i]['classifier_network']['hidden_nac'] = True
    configs[i]['name'] = 'bL16_4xTr_MhaFfn_Ffn(x,y)*y+x_' + \
        'MhaPoolVnLin_MaxPool_4096d_ClassNac_Snli_'+str(i)

task = 'Snli'
if (task != 'Snli') and (task != 'Mnli') and (task != 'Anli'):
    for config in configs:
        config['name'] = config['name'].replace('Snli', task)
        config['classifier_network']['num_classes'] = 2  # PAWS & QNLI == 2

        if task == 'Paws':
            config['training']['lr'] = ([4e-5]*12 + [2e-5]*12 + [1e-5]*12 +
                                        [5e-6]*12 + [2e-6]*6 + [1e-6]*6 +
                                        [5e-7]*6 + [2e-7]*6 + [1e-7]*6)
        elif task == 'Qnli':
            config['training']['lr'] = ([4e-5]*5 + [2e-5]*6 + [1e-5]*5 +
                                        [5e-6]*6 + [2e-6]*5 + [1e-6]*6 +
                                        [5e-7]*6 + [2e-7]*6 + [1e-7]*6)
        config['training']['task'] = task.lower()

if (task == 'Mnli') or (task == 'Anli'):
    for config in configs:
        config['name'] = config['name'].replace('Snli', task)
        config['training']['task'] = task.lower()
