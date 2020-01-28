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


class QnliBatch(tf.keras.utils.Sequence):
# class QnliBatch():
    def __init__(self, config, valid=False):
        """
        Initialize batcher
        :param config:
        """
        self.config = config
        self.valid = valid

        self.train_parts_per_epoch = 12
        self.train_part = 0

        self.datasets_dir = "./datasets/QNLIv2/QNLI/"
        self.batch_dir = './train/datasets/'
        self.labels = ['not_entailment', 'entailment']

        # if not self._batch_file_does_exist():
            # self._create_batch_file()

        self.batch_part = 0

        if len(batch_train_data) == 0:
            self._init_batch()

    def _batch_file_does_exist(self):
        """
        Checks if the batch file was already created.
        :return:    True - batch file exist.
                    False - batch file does not exist.
        """
        batch_files_exist = True
        datasets_files = os.listdir(self.datasets_dir)
        for dataset in datasets_files:
            dataset_name = '.'.join(dataset.split('.')[0:-1])
            batch_files = os.listdir(self.batch_dir)
            for batch in batch_files:
                if dataset_name not in batch:
                    batch_files_exist = False
                    break
        return batch_files_exist

    def _create_batch_file(self):
        """
        Preprocess batch files for faster training
        :return:
        """
        self.embedding = RoBERTaEmbeddings(
            pretrained_model_name_or_path="roberta-large",
            layers="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20," +
                   "21,22,23,24",
            pooling_operation="mean", use_scalar_mix=True)
        self._read_datasets()

    def _read_datasets(self):
        """
        Read QNLI dataset
        :return:
        """
        datasets_files = os.listdir(self.datasets_dir)
        pickle_dataset = './train/datasets/'
        os.makedirs(pickle_dataset, exist_ok=True)
        for dataset in datasets_files:
            if dataset.split('.')[-1] == 'tsv':
                processed_dataset = []
                with open(self.datasets_dir + '/' + dataset) as f:
                    for idx, line in enumerate(f):
                        if idx > 1:
                            data = line.replace('\n', '').split('\t')
                            sent1 = data[1]
                            sent2 = data[2]
                            if (len(sent1.split(' ')) < 200) \
                                and (len(sent2.split(' ')) < 200):
                                label = self.labels.index(data[3])
                                sents_emb = self._process_sentences(
                                    [sent1, sent2])
                                processed_dataset.append({
                                    'sentences_emb': sents_emb,
                                    'label': label
                                })
                            else:
                                print("-----------ERROR---------")
                                print(data)
                                print(len(sent1.split(' ')))
                                print(len(sent2.split(' ')))
                            if (idx % 100) == 0:
                                print(idx)
                    pickle.dump(processed_dataset, open(
                        self.batch_dir + 'qnli_' + dataset.replace(
                            '.tsv', '.pickle'), 'wb'))

    def _process_sentences(self, sentences):
        sentences_emb = []
        for sentence in sentences:
            sentence = " ".join(sentence.split())
            sent = sentence
            if len(sent.strip()) == 0:
                sent = 'empty'
            try:
                sent = Sentence(sent)
                self.embedding.embed(sent)
                sentence_emb = [np.array(t.embedding).astype(np.float16)
                                for t in sent]
                sentences_emb.append(np.array(sentence_emb).astype(np.float16))
            except IndexError:
                print('IndexError')
                print(sentence)
                sentence_emb = [np.array(t.embedding).astype(np.float16)
                                for t in sent]
                sentences_emb.append(np.array(sentence_emb).astype(np.float16))
        sentences_emb_short = sentences_emb
        return sentences_emb_short

    def _init_batch(self):
        """
        Read dataset into memory
        """
        global batch_train_data, batch_valid_data
        batch_train_data = []

        train_files_list = []
        test_file_list = []
        batch_files = os.listdir(self.batch_dir)
        for batch in batch_files:
            if ('train' in batch) and ('qnli' in batch):
                train_files_list.append(batch)
            elif ('dev' in batch) and ('qnli' in batch):
                test_file_list.append(batch)

        batch_train_data = pickle.load(open(self.batch_dir +
                                            train_files_list[0], 'rb'))
        random.shuffle(batch_train_data)
        if len(batch_valid_data) == 0:
            batch_valid_data = pickle.load(
                open(self.batch_dir + test_file_list[0], 'rb'))

    def _process_batch(self, batch):
        return batch

    def on_epoch_end(self):
        self.train_part += 1
        if self.train_part == self.train_parts_per_epoch:
            self.train_part = 0
            self._init_batch()
        print('self.train_part:')
        print(self.train_part)
        print('')

    def __len__(self):
        global batch_train_data, batch_valid_data

        if self.valid:
            return int(len(batch_valid_data) / self.config['batch_size'])
        else:
            # return int(len(batch_train_data) / self.config['batch_size'])
            return int(len(batch_train_data) / self.config['batch_size']
                       / self.train_parts_per_epoch)

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
        label = np.zeros(shape=(self.config['batch_size'], 2),
                         dtype=np.bool_)

        batch_dataset = batch_valid_data if self.valid else batch_train_data

        for b_idx in range(self.config['batch_size']):
            if self.valid:
                d_idx = b_idx + (idx - 1) * self.config['batch_size']
            else:
                d_idx = b_idx + (idx - 1) * self.config['batch_size'] + \
                        self.train_part * int(len(batch_train_data)
                                              / self.train_parts_per_epoch)

            sent1 = batch_dataset[d_idx]['sentences_emb'][0]
            sent2 = batch_dataset[d_idx]['sentences_emb'][1]
            qnli_label = batch_dataset[d_idx]['label']

            sentence1[b_idx][0:min(len(sent1), self.config['max_sent_len'])] =\
                sent1[0:min(len(sent1), self.config['max_sent_len'])]
            sentence1_mask[b_idx][0:min(len(sent1),
                                        self.config['max_sent_len'])] = True

            sentence2[b_idx][0:min(len(sent2), self.config['max_sent_len'])] =\
                sent2[0:min(len(sent2), self.config['max_sent_len'])]
            sentence2_mask[b_idx][0:min(len(sent2),
                                        self.config['max_sent_len'])] = True

            label[b_idx][qnli_label] = True

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
    batcher = QnliBatch({
        'batch_size': 2,
        'max_sent_cnt': 6,
        'max_sent_len': 64,
        'word_edim': 1024
    })
    batch_x, batch_y = batcher.__getitem__(1)
    print(batch_x)


# test()
