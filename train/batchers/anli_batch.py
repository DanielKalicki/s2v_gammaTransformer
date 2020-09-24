import numpy as np
import os.path
import pickle
import tensorflow as tf
from typing import Optional
import random
random.seed(0)
# from flair.embeddings import RoBERTaEmbeddings
# from flair.data import Sentence

batch_train_data = []
batch_valid_data = []


class AnliBatch(tf.keras.utils.Sequence):
    def __init__(self, config, valid=False):
        """
        Initialize batcher
        :param config:
        """
        self.config = config
        self.valid = valid

        self.batch_dir = './train/datasets/'
        self.labels = ['contradiction', 'neutral', 'entailment']

        self.train_batch_part = -1
        # self.train_batch_part = 8
        self._init_batch()

    def _init_batch(self):
        """
        Read dataset into memory
        """
        global batch_train_data, batch_valid_data
        batch_train_data = []

        snli_train_files_list = []
        snli_test_file_list = []
        mnli_train_files_list = []
        mnli_test_file_list = []
        batch_files = os.listdir(self.batch_dir)
        for batch in batch_files:
            if ('train' in batch) and ('snli' in batch):
                snli_train_files_list.append(batch)
            elif ('test' in batch) and ('snli' in batch):
                snli_test_file_list.append(batch)
            elif ('train' in batch) and ('mnli' in batch):
                mnli_train_files_list.append(batch)
            elif ('mismatched' in batch) and ('multinli' in batch):
                mnli_test_file_list.append(batch)

        self.train_batch_part += 1
        if self.train_batch_part >= len(snli_train_files_list):
            self.train_batch_part = 0
        print(self.train_batch_part)

        snli_train_files_list.sort()
        mnli_train_files_list.sort()
        snli_train_file = snli_train_files_list[self.train_batch_part]
        mnli_train_file = mnli_train_files_list[self.train_batch_part]
        print(snli_train_file)
        print(mnli_train_file)
        print("")

        batch_train_data = pickle.load(
            open(self.batch_dir + snli_train_file, 'rb'))
        batch_train_data.extend(pickle.load(
            open(self.batch_dir + mnli_train_file, 'rb')))
        random.shuffle(batch_train_data)
        # batch_train_data = batch_train_data[0:17136]
        if len(batch_valid_data) == 0:
            batch_valid_data = pickle.load(
                open(self.batch_dir + snli_test_file_list[0], 'rb'))
            batch_valid_data.extend(pickle.load(
                open(self.batch_dir + mnli_test_file_list[0], 'rb')))

    def on_epoch_end(self):
        self._init_batch()

    def __len__(self):
        global batch_train_data, batch_valid_data

        if self.valid:
            return int(len(batch_valid_data) / self.config['batch_size'])
        else:
            return int(len(batch_train_data) / self.config['batch_size'])

    def __getitem__(self, idx):
        """
        Returns batch.
        """
        global batch_train_data, batch_valid_data

        sentence1 = np.zeros(shape=(self.config['batch_size'],
                                    self.config['max_sent_len'],
                                    self.config['word_edim']),
                             dtype=np.float32)
        sentence1_mask = np.zeros(shape=(self.config['batch_size'],
                                         self.config['max_sent_len']),
                                  dtype=np.bool_)
        sentence2 = np.zeros(shape=(self.config['batch_size'],
                                    self.config['max_sent_len'],
                                    self.config['word_edim']),
                             dtype=np.float32)
        sentence2_mask = np.zeros(shape=(self.config['batch_size'],
                                         self.config['max_sent_len']),
                                  dtype=np.bool_)
        label = np.zeros(shape=(self.config['batch_size'], 3),
                         dtype=np.bool_)

        batch_dataset = batch_valid_data if self.valid else batch_train_data

        for b_idx in range(self.config['batch_size']):
            d_idx = b_idx + (idx - 1) * self.config['batch_size']

            sent1 = batch_dataset[d_idx]['sentences_emb'][0]
            sent2 = batch_dataset[d_idx]['sentences_emb'][1]
            nli_label = batch_dataset[d_idx]['label']

            sentence1[b_idx][0:min(len(sent1), self.config['max_sent_len'])] =\
                sent1[0:min(len(sent1), self.config['max_sent_len'])]
            sentence1_mask[b_idx][0:min(len(sent1),
                                        self.config['max_sent_len'])] = True

            sentence2[b_idx][0:min(len(sent2), self.config['max_sent_len'])] =\
                sent2[0:min(len(sent2), self.config['max_sent_len'])]
            sentence2_mask[b_idx][0:min(len(sent2),
                                        self.config['max_sent_len'])] = True

            label[b_idx][nli_label] = True

        x = {
            'sentence1': sentence1,
            'sentence1_mask': sentence1_mask,
            'sentence1_transformer_mask': create_attention_mask(
                sentence1_mask, is_causal=False, bert_attention=True),
            'sentence2': sentence2,
            'sentence2_mask': sentence2_mask,
            'sentence2_transformer_mask': create_attention_mask(
                sentence2_mask, is_causal=False, bert_attention=True),
        }
        y = {
            'nli_classifier_model': label,
        }
        return x, y


def create_attention_mask(pad_mask: Optional[np.array], is_causal: bool,
                          batch_size: Optional[int] = None,
                          length: Optional[int] = None,
                          bert_attention: bool = False) -> np.array:
    ndim = pad_mask.ndim
    pad_shape = pad_mask.shape
    if ndim == 3:
        pad_mask = np.reshape(pad_mask, (pad_shape[0]*pad_shape[1],
                                         pad_shape[2]))
    if pad_mask is not None:
        assert pad_mask.ndim == 2
        batch_size, length = pad_mask.shape
    if is_causal:
        b = np.cumsum(np.eye(length, dtype=np.float32), axis=0)
    else:
        b = np.ones((length, length), dtype=np.float32)
    b = np.reshape(b, [1, 1, length, length])
    b = np.repeat(b, batch_size, axis=0)  # B, 1, L, L
    if pad_mask is not None:
        _pad_mask = pad_mask[..., np.newaxis]
        _pad_mask = np.repeat(_pad_mask, length, 2)
        _pad_mask_t = np.transpose(_pad_mask, [0, 2, 1])
        if bert_attention:
            tmp = _pad_mask_t
        else:
            tmp = _pad_mask * _pad_mask_t
        tmp = tmp[:, np.newaxis, ...]
        if b is None:
            b = tmp.astype(np.float32)
        else:
            b = b * tmp
    if ndim == 3:
        b_shape = b.shape
        b = np.reshape(b, (pad_shape[0], pad_shape[1], b_shape[1],
                           b_shape[2], b_shape[3]))
    return b


def test():
    batcher = AnliBatch({
        'batch_size': 2,
        'max_sent_cnt': 6,
        'max_sent_len': 64,
        'word_edim': 1024
    })
    batch_x, batch_y = batcher.__getitem__(1)
    print(batch_x['sentences'][0])


# test()
