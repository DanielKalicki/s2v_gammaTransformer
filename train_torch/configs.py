import tensorflow as tf
import copy

default_config = {
    'batch_size': 16,
    'max_sent_len': 64,
    'word_edim': 1024,
    's2v_dim': 4096,
    'name': '',
    'restore_name': '',

    'sentence_encoder': {
        'input_drop': 0.0,
        'input_gaussian_noise': 0.0,
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
        'in_dropout': 0.0,  # input dropout
        'dropout': 0.0,  # hidden dropout
        'hidden_layer_type': 'dense',  # dense, nac, nalu
        'hidden_layer_norm': False,
        'hidden_activation': 'gelu',
        'prediction_layer_type': 'dense',  # dense, nac, nalu
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
        'label_noise': 0.0,
        'epochs': 300,
        'log': True
    }
}

configs = []
for i in range(100):
    configs.append(copy.deepcopy(default_config))

# -----------------------------------------------------------------------------
for i in range(45, 46):
    (configs[i]['sentence_encoder']['transformer']['mha_modifications']
               ['output_mha']) = True
    (configs[i]['sentence_encoder']['transformer']['mha_modifications']
               ['output_mha_num_heads']) = 16
    configs[i]['sentence_encoder']['pooling']['mha']['input_ffn'] = 'mha'
    configs[i]['sentence_encoder']['transformer']['gate_type'] = \
        'Ffn(x,y)*y + x'
    configs[i]['classifier_network']['hidden_layer_type'] = 'nac'
    configs[i]['classifier_network']['prediction_layer_type'] = 'nac'
    configs[i]['training']['loss_mean0_s2v'] = True
    configs[i]['max_sent_len'] = 256
    configs[i]['batch_size'] = 64
    configs[i]['name'] = 'bL4_4xTr_MhaMha16hFfn_Ffn(x,y)*y+x_' + \
        'MhaPoolVmha_MaxPool_4096d_Snli_ClassNac_s2vLoss_valanliloss_'+str(i)
# -----------------------------------------------------------------------------
for i in range(46, 47):
    configs[i]['sentence_encoder']['transformer']['gate_type'] = \
        'Ffn(x,y)*y + x'
    configs[i]['sentence_encoder']['pooling']['mha']['inner_dim'] = 1024
    configs[i]['max_sent_len'] = 64
    configs[i]['batch_size'] = 32+8
    configs[i]['training']['optimizer'] = 'Adam'
    configs[i]['training']['label_smoothing'] = 0.0
    configs[i]['training']['clipnorm'] = -1.0
    configs[i]['training']['lr'] = 4e-5
    configs[i]['name'] = 'bL40_sl64_4xTr_Mha16hDense_Ffn(x,y)*y+x_' + \
        'MhaPool_MaxPool_1024d_Snli_valanliloss_'+str(i)
# -----------------------------------------------------------------------------
for i in range(47, 48):
    configs[i]['sentence_encoder']['transformer']['gate_type'] = \
        'Ffn(x,y)*y + x'
    configs[i]['sentence_encoder']['pooling']['mha']['inner_dim'] = 1024
    configs[i]['max_sent_len'] = 64
    configs[i]['batch_size'] = 32+8
    configs[i]['training']['optimizer'] = 'Adam'
    configs[i]['training']['label_smoothing'] = 0.0
    configs[i]['training']['clipnorm'] = -1.0
    configs[i]['training']['lr'] = 8e-5
    configs[i]['name'] = 'bL40_sl64_4xTr_Mha16hDense_Ffn1xH(x,y)*y+x_' + \
        'MhaPool_MaxPool_1024d_Snli_Lr8e-5dec0.85_valanliloss_'+str(i)
# -----------------------------------------------------------------------------
for i in range(48, 49):
    configs[i]['sentence_encoder']['transformer']['gate_type'] = \
        'Ffn(x,y)*y + x'
    configs[i]['sentence_encoder']['pooling']['mha']['inner_dim'] = 1024
    configs[i]['max_sent_len'] = 64
    configs[i]['batch_size'] = 16
    configs[i]['training']['optimizer'] = 'Adam'
    configs[i]['training']['label_smoothing'] = 0.0
    configs[i]['training']['clipnorm'] = -1.0
    configs[i]['training']['lr'] = 8e-5
    configs[i]['name'] = 'bL16_sl64_4xTrGelu_Mha16hDense_Ffn1xH(x,y)*y+x_' + \
        'MaxPool_1024d_Snli_Lr8e-5dec0.85_valanliloss_'+str(i)
# -----------------------------------------------------------------------------
for i in range(49, 50):
    configs[i]['sentence_encoder']['transformer']['gate_type'] = \
        'Ffn(x,y)*y + x'
    configs[i]['sentence_encoder']['pooling']['mha']['inner_dim'] = 1024
    configs[i]['max_sent_len'] = 64
    configs[i]['batch_size'] = 16
    configs[i]['training']['optimizer'] = 'Adam'
    configs[i]['training']['label_smoothing'] = 0.0
    configs[i]['training']['clipnorm'] = -1.0
    configs[i]['training']['lr'] = 8e-5
    configs[i]['name'] = 'bL16_sl64_4xTrNorm_Mha16hDense_Ffn1xH(x,y)*y+x_' + \
        'MaxPool_1024d_Snli_Lr8e-5dec0.85_valanliloss_'+str(i)
# -----------------------------------------------------------------------------
for i in range(50, 51):
    configs[i]['sentence_encoder']['transformer']['gate_type'] = \
        'Ffn(x,y)*y + x'
    configs[i]['sentence_encoder']['pooling']['mha']['inner_dim'] = 1024
    configs[i]['sentence_encoder']['transformer']['num_layers'] = 6
    configs[i]['max_sent_len'] = 64
    configs[i]['batch_size'] = 16
    configs[i]['training']['optimizer'] = 'Adam'
    configs[i]['training']['label_smoothing'] = 0.0
    configs[i]['training']['clipnorm'] = -1.0
    configs[i]['training']['lr'] = 8e-5
    configs[i]['name'] = 'bL16_sl64_6xTr_Mha16hDense_Ffn1xH(x,y)*y+x_' + \
        'MaxPool_1024d_Snli_Lr8e-5dec0.85_valanliloss_'+str(i)
# -----------------------------------------------------------------------------
for i in range(51, 52):
    configs[i]['sentence_encoder']['transformer']['gate_type'] = \
        'Ffn(x,y)*y + x'
    configs[i]['sentence_encoder']['pooling']['mha']['inner_dim'] = 1024
    configs[i]['sentence_encoder']['transformer']['num_layers'] = 4
    configs[i]['max_sent_len'] = 64
    configs[i]['batch_size'] = 16
    configs[i]['training']['optimizer'] = 'Adam'
    configs[i]['training']['label_smoothing'] = 0.0
    configs[i]['training']['clipnorm'] = -1.0
    configs[i]['training']['lr'] = 8e-5
    configs[i]['name'] = 'bL16_sl64_2x(4xTr)_Mha16hDense_Ffn1xH(x,y)*y+x_' + \
        'MaxPool_1024d_Snli_Lr8e-5dec0.85_valanliloss_'+str(i)
# -----------------------------------------------------------------------------
for i in range(52, 53):
    configs[i]['sentence_encoder']['transformer']['gate_type'] = \
        'Ffn(x,y)*y + x'
    configs[i]['sentence_encoder']['pooling']['mha']['inner_dim'] = 4*1024
    configs[i]['sentence_encoder']['transformer']['num_layers'] = 4
    configs[i]['max_sent_len'] = 32
    configs[i]['batch_size'] = 16
    configs[i]['training']['optimizer'] = 'Adam'
    configs[i]['training']['label_smoothing'] = 0.0
    configs[i]['training']['clipnorm'] = -1.0
    configs[i]['training']['lr'] = 8e-5
    configs[i]['name'] = 'bL16_sl32_4xTr_Mha16hDense_Ffn1xH(x,y)*y+x_' + \
        'Mha4xPool_MaxPool_1024d_Snli_Lr8e-5dec0.96_valanliloss_epoch4xFile.1LenFix2_'+str(i)
# -----------------------------------------------------------------------------
for i in range(53, 54):
    configs[i]['sentence_encoder']['input_drop'] = 0.1
    configs[i]['sentence_encoder']['transformer']['gate_type'] = \
        'Ffn(x,y)*y + x'
    configs[i]['sentence_encoder']['pooling']['mha']['inner_dim'] = 4*1024
    configs[i]['sentence_encoder']['transformer']['num_layers'] = 4
    configs[i]['sentence_encoder']['transformer']['residual_dropout'] = 0.1
    configs[i]['classifier_network']['in_dropout'] = 0.1
    configs[i]['max_sent_len'] = 32
    configs[i]['batch_size'] = 16
    configs[i]['training']['optimizer'] = 'Adam'
    configs[i]['training']['label_smoothing'] = 0.0
    configs[i]['training']['clipnorm'] = -1.0
    configs[i]['training']['lr'] = 8e-5
    configs[i]['name'] = 'bL16_sl32_InDr.1_4xTrDr.1_Mha16hDense_Ffn1xH(x,y)*y+x_' + \
        'Mha4xPool_MaxPool_dr.1_1024d_Snli_Lr8e-5dec0.96_valanliloss_epoch4xFile.1LenFix2_'+str(i)
# -----------------------------------------------------------------------------
for i in range(54, 55):
    configs[i]['sentence_encoder']['input_drop'] = 0.0
    configs[i]['sentence_encoder']['transformer']['gate_type'] = \
        'Ffn(x,y)*y + x'
    configs[i]['sentence_encoder']['pooling']['mha']['inner_dim'] = 4*1024
    configs[i]['sentence_encoder']['transformer']['num_layers'] = 4
    configs[i]['sentence_encoder']['transformer']['residual_dropout'] = 0.0
    configs[i]['sentence_encoder']['transformer']['ffn_dim'] = 256
    configs[i]['classifier_network']['in_dropout'] = 0.0
    configs[i]['max_sent_len'] = 32
    configs[i]['batch_size'] = 16
    configs[i]['training']['optimizer'] = 'Adam'
    configs[i]['training']['label_smoothing'] = 0.0
    configs[i]['training']['clipnorm'] = -1.0
    configs[i]['training']['lr'] = 8e-5
    configs[i]['name'] = 'bL16_sl32_4xTr_Mha16hDense_Ffnd4_Ffn1xH(x,y)*y+x_' + \
        'Mha4xPool_MaxPool_1024d_Snli_Lr8e-5dec0.96_valanliloss_epoch4xFile.1LenFix2_'+str(i)
# -----------------------------------------------------------------------------
for i in range(55, 56):
    configs[i]['sentence_encoder']['input_drop'] = 0.0
    configs[i]['sentence_encoder']['transformer']['gate_type'] = \
        'Ffn(x,y)*y + x'
    configs[i]['sentence_encoder']['pooling']['mha']['inner_dim'] = 4*1024
    configs[i]['sentence_encoder']['transformer']['num_layers'] = 4
    configs[i]['sentence_encoder']['transformer']['residual_dropout'] = 0.0
    configs[i]['sentence_encoder']['transformer']['ffn_dim'] = 256
    configs[i]['classifier_network']['in_dropout'] = 0.0
    configs[i]['max_sent_len'] = 32
    configs[i]['batch_size'] = 16
    configs[i]['training']['optimizer'] = 'Adam'
    configs[i]['training']['label_smoothing'] = 0.0
    configs[i]['training']['clipnorm'] = -1.0
    configs[i]['training']['lr'] = 8e-5
    configs[i]['name'] = 'bL16_sl32_4xTr_DextraMha16hDense_Ffnd4_Ffn1xH(x,y)*y+x_' + \
        'Mha4xPool_MaxPool_1024d_Snli_Lr8e-5dec0.96_valanliloss_epoch4xFile.1LenFix2_'+str(i)
# -----------------------------------------------------------------------------
for i in range(56, 57):
    configs[i]['sentence_encoder']['input_drop'] = 0.0
    configs[i]['sentence_encoder']['transformer']['gate_type'] = \
        'Ffn(x,y)*y + x'
    configs[i]['sentence_encoder']['pooling']['mha']['inner_dim'] = 4*1024
    configs[i]['sentence_encoder']['transformer']['num_layers'] = 6
    configs[i]['sentence_encoder']['transformer']['residual_dropout'] = 0.0
    configs[i]['sentence_encoder']['transformer']['ffn_dim'] = 256
    configs[i]['classifier_network']['in_dropout'] = 0.0
    configs[i]['max_sent_len'] = 32
    configs[i]['batch_size'] = 16
    configs[i]['training']['optimizer'] = 'Adam'
    configs[i]['training']['label_smoothing'] = 0.0
    configs[i]['training']['clipnorm'] = -1.0
    configs[i]['training']['lr'] = 8e-5
    configs[i]['name'] = 'bL16_sl32_6xTrNorm_SubLay.25_DextraMha16hDense_Ffnd4_Ffn1xH(x,y)*y+x_' + \
        'Mha4xPool_MaxPool_1024d_Snli_Lr8e-5dec0.96_valanliloss_epoch4xFile.1LenFix2_'+str(i)
# -----------------------------------------------------------------------------
for i in range(57, 58):
    configs[i]['sentence_encoder']['input_drop'] = 0.0
    configs[i]['sentence_encoder']['transformer']['gate_type'] = \
        'Ffn(x,y)*y + x'
    configs[i]['sentence_encoder']['pooling']['mha']['inner_dim'] = 4*1024
    configs[i]['sentence_encoder']['transformer']['num_layers'] = 8
    configs[i]['sentence_encoder']['transformer']['residual_dropout'] = 0.0
    configs[i]['sentence_encoder']['transformer']['ffn_dim'] = 256
    configs[i]['classifier_network']['in_dropout'] = 0.0
    configs[i]['max_sent_len'] = 32
    configs[i]['batch_size'] = 16
    configs[i]['training']['optimizer'] = 'Adam'
    configs[i]['training']['label_smoothing'] = 0.0
    configs[i]['training']['clipnorm'] = -1.0
    configs[i]['training']['lr'] = 8e-5
    configs[i]['name'] = 'bL16_sl32_8xTrNorm_SubLay.25_Mha16hDense_Ffnd4_Ffn4dH(x,y)*y+x_' + \
        'Mha4xPool_MaxPool_1024d_Snli_Lr8e-5dec0.96_valanliloss_epoch4xFile.1LenFix2_'+str(i)
# -----------------------------------------------------------------------------
for i in range(58, 59):
    configs[i]['sentence_encoder']['input_drop'] = 0.0
    configs[i]['sentence_encoder']['transformer']['gate_type'] = \
        'Ffn(x,y)*y + x'
    configs[i]['sentence_encoder']['pooling']['mha']['inner_dim'] = 4*1024
    configs[i]['sentence_encoder']['transformer']['num_layers'] = 6
    configs[i]['sentence_encoder']['transformer']['residual_dropout'] = 0.0
    configs[i]['sentence_encoder']['transformer']['ffn_dim'] = 256
    configs[i]['classifier_network']['in_dropout'] = 0.0
    configs[i]['max_sent_len'] = 32
    configs[i]['batch_size'] = 16
    configs[i]['training']['optimizer'] = 'Adam'
    configs[i]['training']['label_smoothing'] = 0.0
    configs[i]['training']['clipnorm'] = -1.0
    configs[i]['training']['lr'] = 8e-5
    configs[i]['name'] = 'bL16_sl32_6xTrNorm_SubLay.25_Mha16hDense_Ffnd4_Ffn1xH(x,y)*y+x_' + \
        'Mha4xPool__MaxPoolMaskFix_4096d_Snli_Lr8e-5dec0.96_valanliloss_epoch4xFile.1LenFix2_'+str(i)
# -----------------------------------------------------------------------------
for i in range(59, 60):
    configs[i]['sentence_encoder']['input_drop'] = 0.0
    configs[i]['sentence_encoder']['transformer']['gate_type'] = \
        'Ffn(x,y)*y + x'
    configs[i]['sentence_encoder']['pooling']['mha']['inner_dim'] = 4*1024
    configs[i]['sentence_encoder']['transformer']['num_layers'] = 6
    configs[i]['sentence_encoder']['transformer']['residual_dropout'] = 0.0
    configs[i]['sentence_encoder']['transformer']['ffn_dim'] = 256
    configs[i]['sentence_encoder']['pooling']['mha']['num_heads'] = 128
    configs[i]['classifier_network']['in_dropout'] = 0.0
    configs[i]['max_sent_len'] = 32
    configs[i]['batch_size'] = 16
    configs[i]['training']['optimizer'] = 'Adam'
    configs[i]['training']['label_smoothing'] = 0.0
    configs[i]['training']['clipnorm'] = -1.0
    configs[i]['training']['lr'] = 8e-5
    # configs[i]['name'] = 'bL16_sl32_CeLoss.TestWordsNoise.3_6xTr_SubLay.25_Mha16hDense_Ffnd4_Ffn1xH(x,y)*y+x_' + \
    configs[i]['name'] = 'bL16_sl32_CeLoss.TestWordsNoise.3.Noisex0.1CosSimv1_6xTr_TanhSubLay.25_Mha16hDense_Ffnd4_Ffn1xH(x,y)*y+x_' + \
        'Mha128h4xPool_MaxPoolMaskFix_4096d_Snli_Lr8e-5dec0.96.1LenFix2_'+str(i)


task = 'Anli'
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
