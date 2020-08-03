import numpy as np
import os.path
import pickle
import tensorflow as tf
from typing import Optional
import random
random.seed(0)
import torch
from torch.utils.data import Dataset
# from flair.embeddings import RoBERTaEmbeddings
# from flair.data import Sentence

batch_train_data = []
batch_valid_data = []


class AnliBatch(Dataset):
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
            return len(batch_valid_data)
        else:
            return len(batch_train_data)

    def __getitem__(self, idx):
        """
        Returns batch.
        """
        global batch_train_data, batch_valid_data

        if torch.is_tensor(idx):
            idx = idx.tolist()

        sentence1 = torch.zeros(shape=(self.config['max_sent_len'], self.config['word_edim']),
                                dtype=torch.float)
        sentence1_mask = torch.full(shape=(self.config['max_sent_len']), fill_value=float("-inf"),
                                    dtype=np.float)
        sentence2 = torch.zeros(shape=(self.config['max_sent_len'], self.config['word_edim']),
                                dtype=torch.float)
        sentence2_mask = torch.full(shape=(self.config['max_sent_len']), fill_value=float("-inf"),
                                    dtype=np.float)
        label = torch.zeros(shape=(3), dtype=np.float)

        batch_dataset = batch_valid_data if self.valid else batch_train_data

        sent1 = batch_dataset[idx]['sentences_emb'][0]
        sent2 = batch_dataset[idx]['sentences_emb'][1]
        nli_label = batch_dataset[idx]['label']

        sentence1[0:min(len(sent1), self.config['max_sent_len'])] =\
            sent1[0:min(len(sent1), self.config['max_sent_len'])]
        sentence1_mask[0:min(len(sent1), self.config['max_sent_len'])] = torch.tensor(0.0)

        sentence2[0:min(len(sent2), self.config['max_sent_len'])] =\
            sent2[0:min(len(sent2), self.config['max_sent_len'])]
        sentence2_mask[0:min(len(sent2), self.config['max_sent_len'])] = torch.tensor(0.0)

        label[nli_label] = 1.0

        return sentence1, sentence1_mask, sentence2, sentence2_mask, label

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
