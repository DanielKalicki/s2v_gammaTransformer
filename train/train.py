import tensorflow as tf
import os
import shutil

from batchers.snli_batch import SnliBatch
from batchers.mnli_batch import MnliBatch
from batchers.anli_batch import AnliBatch
from batchers.qnli_batch import QnliBatch
from batchers.paws_batch import PawsBatch
from models.sentence_encoder_model import SentenceEncoderModel
from models.nli_matching_model import NliClassifierModel
from tensorflow.keras.callbacks import (TensorBoard, LearningRateScheduler,
                                        ModelCheckpoint)
from configs import configs
import sys

# tf.executing_eagerly()

# -----------------------------------------------------------------------------
# Model configuration
# -----------------------------------------------------------------------------
print(int(sys.argv[1]))
config = configs[int(sys.argv[1])]

# -----------------------------------------------------------------------------
# Save model files
# -----------------------------------------------------------------------------
if not os.path.exists('train/logs'):
    os.mkdir('train/logs')
os.mkdir('train/logs/'+config['name'])
shutil.copytree('train/batchers', 'train/logs/' + config['name'] + '/batchers')
shutil.copytree('train/models', 'train/logs/' + config['name'] + '/models')
shutil.copytree('train/nlp_blocks',
                'train/logs/' + config['name'] + '/nlp_blocks')
shutil.copyfile('train/train.py', 'train/logs/' + config['name'] + '/train.py')
shutil.copyfile('train/configs.py', 'train/logs/' + config['name'] +
                '/configs.py')

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
nli_matching_model = NliClassifierModel(config)
nli_predictions = nli_matching_model(sent1_s2v, sent2_s2v)

model = tf.keras.models.Model(inputs=[input_sentence1, input_sentence1_mask,
                                      input_sentence1_transformer_mask,
                                      input_sentence2, input_sentence2_mask,
                                      input_sentence2_transformer_mask],
                              outputs=[nli_predictions])
if config['restore_name'] != '':
    model.load_weights("./train/save/"+config['restore_name']+"/model")

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
elif config['training']['optimizer'] == 'SGD':
    opt = tf.keras.optimizers.SGD(lr=config['training']['lr'][0],
                                   clipnorm=config['training']['clipnorm'])

# -----------------------------------------------------------------------------
# Batchers
# -----------------------------------------------------------------------------
if config['training']['task'] == 'snli':
    generator_train = SnliBatch(config)
    generator_valid = SnliBatch(config, valid=True)
elif config['training']['task'] == 'qnli':
    generator_train = QnliBatch(config)
    generator_valid = QnliBatch(config, valid=True)
elif config['training']['task'] == 'paws':
    generator_train = PawsBatch(config)
    generator_valid = PawsBatch(config, valid=True)
elif config['training']['task'] == 'mnli':
    generator_train = MnliBatch(config)
    generator_valid = MnliBatch(config, valid=True)
elif config['training']['task'] == 'anli':
    generator_train = AnliBatch(config)
    generator_valid = AnliBatch(config, valid=True)

# -----------------------------------------------------------------------------
# Training callbacks
# -----------------------------------------------------------------------------
def loss(y_true, y_pred):
    label_smoothing = config['training']['label_smoothing']
    new_onehot_labels = y_true * (1 - label_smoothing) + label_smoothing / 2
    loss = tf.keras.losses.categorical_crossentropy(new_onehot_labels, y_pred)
    if config['training']['loss_mean0_s2v']:
        v1 = tf.reduce_sum(tf.math.abs(tf.reduce_mean(sent1_s2v, axis=0)))
        v2 = tf.reduce_sum(tf.math.abs(tf.reduce_mean(sent2_s2v, axis=0)))
        loss += (v1+v2)*1e-5
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
                             monitor='val_loss', verbose=1, save_best_only=True,
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
