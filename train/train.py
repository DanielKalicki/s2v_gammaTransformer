import tensorflow as tf
import os
import shutil

from batchers.snli_batch import SnliBatch
from models.sentence_encoder_model import SentenceEncoderModel
from models.nli_matching_model import NliClassifierModel
from tensorflow.keras.callbacks import (TensorBoard, LearningRateScheduler,
                                        ModelCheckpoint)

# -----------------------------------------------------------------------------
# Model configuration
# -----------------------------------------------------------------------------
config = {
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
            # TODO Mha with FFN
        },
        'pooling': {
            'pooling_method': 'mha',  # ['mha', 'dense']
            # 'dense': {
            #     'layer_cnt': 1,
            #     'hidden_activation': 'gelu'
            #     'output_dim': 4096,
            # }
            'mha': {
                'inner_dim': 4096,
                'num_heads': 32,
                'attention_dropout': 0.0,
                'output_projection': False,
                'output_dim': 4096,  # used only if output_projection = True
                # TODO Non linearity in v (2 layers with gelu)
            },
            'pooling_activation': None,  # activation function used before pool
            'pooling_function': 'mean',  # ['mean', 'max']
        },
        'pool_projection': False,  # if True pooling output will be projected
                                   #    to s2v_dim
    },

    'classifier_network': {
        'hidden_dim': 512,
        'hidden_layer_cnt': 1,
        'dropout': 0.0,
        'hidden_activation': 'gelu',
        'prediction_activation': tf.keras.activations.softmax,
    },

    'training': {
        'optimizer': 'Nadam',
        'clipnorm': 1.,
        'lr': ([4e-5]*5 + [2e-5]*6 + [1e-5]*5 + [5e-6]*6 + [2e-6]*5 + [1e-6]*4 + [5e-7]*3
               + [2e-7]*3 + [1e-7]*3),
        'label_smoothing': 0.2,
        'label_noise': 0.0
    }
}

config['name'] = '_bLoneT' + str(config['batch_size']) + \
                 '_4xTr_s2v4096d_TrMhaWithFfn1LxIn1MhaX1_mhaIntern4k' + \
                 '_noCProjAttnInMhaPool_gDenseXY' + \
                 '_noS2vDense_MhaPoolVnLinearX1_snlifullfixShuffle_diffLrSch5'

# sent2vec_GatedModifiedMhaTransformer
# sent2vec_γTransformer

# -----------------------------------------------------------------------------
# Save model files
# -----------------------------------------------------------------------------
os.mkdir('train/logs/'+config['name'])
shutil.copytree('train/batchers', 'train/logs/' + config['name'] + '/batchers')
shutil.copytree('train/models', 'train/logs/' + config['name'] + '/models')
shutil.copytree('train/nlp_blocks',
                'train/logs/' + config['name'] + '/nlp_blocks')
shutil.copyfile('train/train.py', 'train/logs/' + config['name'] + '/train.py')

# -----------------------------------------------------------------------------
# Model inputs
# -----------------------------------------------------------------------------
input_sentence1 = tf.keras.layers.Input(
    shape=(config['max_sent_len'], config['word_edim'],),
    name='sentence1')
input_sentence1_mask = tf.keras.layers.Input(
    shape=(config['max_sent_len'],),
    name='sentence1_mask')
input_sentence1_transformer_mask = tf.keras.layers.Input(
    shape=(1, config['max_sent_len'], config['max_sent_len'],),
    name='sentence1_transformer_mask')
input_sentence2 = tf.keras.layers.Input(
    shape=(config['max_sent_len'], config['word_edim'],),
    name='sentence2')
input_sentence2_mask = tf.keras.layers.Input(
    shape=(config['max_sent_len'],), name='sentence2_mask')
input_sentence2_transformer_mask = tf.keras.layers.Input(
    shape=(1, config['max_sent_len'], config['max_sent_len'],),
    name='sentence2_transformer_mask')

# -----------------------------------------------------------------------------
# Sentence encoder
# -----------------------------------------------------------------------------
sentence_encoder_model = SentenceEncoderModel(config)
sent1_s2v = sentence_encoder_model(input_sentence1, input_sentence1_mask,
                                   input_sentence1_transformer_mask)
sent2_s2v = sentence_encoder_model(input_sentence2, input_sentence2_mask,
                                   input_sentence2_transformer_mask)

# -----------------------------------------------------------------------------
# Classifier
# -----------------------------------------------------------------------------
# paws_matching_model = PawsClassifierModel(config)
nli_matching_model = NliClassifierModel(config)
nli_predictions = nli_matching_model(sent1_s2v, sent2_s2v)

model = tf.keras.models.Model(inputs=[input_sentence1, input_sentence1_mask,
                                      input_sentence1_transformer_mask,
                                      input_sentence2, input_sentence2_mask,
                                      input_sentence2_transformer_mask],
                              outputs=[nli_predictions])

# Restore previous model
# model_name = 'bLoneT16_test10_4xTr_s2v4096d_TrMhaWithFfn_' + \
#             'mhaIntern2kOut4k_noS2vDense_snlifull'
# model.load_weights("train/save/"+model_name+'/model')

# -----------------------------------------------------------------------------
# Optimizer
# -----------------------------------------------------------------------------
opt = None
if config['training']['optimizer'] == 'Nadam':
    opt = tf.keras.optimizers.Nadam(lr=config['training']['lr'][0],
                                    clipnorm=config['training']['clipnorm'])
elif config['training']['optimizer'] == 'Adam':
    opt = tf.keras.optimizers.Adam(lr=config['training']['lr'][0],
                                   clipnorm=config['training']['clipnorm'])

# -----------------------------------------------------------------------------
# Batchers
# -----------------------------------------------------------------------------
generator_train = SnliBatch(config)
generator_valid = SnliBatch(config, valid=True)

# -----------------------------------------------------------------------------
# Training callbacks
# -----------------------------------------------------------------------------
def loss(y_true, y_pred):
    label_smoothing = config['training']['label_smoothing']
    new_onehot_labels = y_true * (1 - label_smoothing) + label_smoothing / 2
    loss = tf.keras.losses.categorical_crossentropy(new_onehot_labels, y_pred)
    return loss


def acc(y_true, y_pred):
    acc = tf.keras.metrics.categorical_accuracy(y_true, y_pred)
    return acc


def step_decay(epoch):
    print('epoch:'+str(epoch)+' lr:'+str(config['training']['lr'][epoch]))
    return config['training']['lr'][epoch]


class BatcherCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs=None):
        print(' ')
        print(config['name'])
        generator_train.on_epoch_end()


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
model.compile(optimizer=opt,
              loss=loss,
              metrics=[acc])
model.summary()

tensorboard = TensorBoard(log_dir='train/logs/'+config['name'],
                          write_graph=False, update_freq='batch')
lrate = LearningRateScheduler(step_decay)
checkpoint = ModelCheckpoint(filepath="train/save/"+config['name']+"/model",
                             monitor='val_acc', verbose=1, save_best_only=True,
                             save_weights_only=True, mode='auto', period=1)

model.fit_generator(generator=generator_train,
                    epochs=len(config['training']['lr']),
                    workers=0,
                    max_queue_size=1,
                    validation_freq=1,
                    validation_data=generator_valid,
                    shuffle=False,
                    callbacks=[tensorboard, lrate, checkpoint,
                               BatcherCallback()])
