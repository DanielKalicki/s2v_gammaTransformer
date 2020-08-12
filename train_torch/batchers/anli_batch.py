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

        if not self.valid:
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
            if self.train_batch_part >= len(snli_train_files_list)//3:
                self.train_batch_part = 0
            print(self.train_batch_part)

            snli_train_files_list.sort()
            mnli_train_files_list.sort()
            batch_train_data = []
            for ifile in range(4):
                snli_train_file = None
                try:
                    snli_train_file = snli_train_files_list[self.train_batch_part*4 + ifile]
                except IndexError:
                    pass
                mnli_train_file = mnli_train_files_list[self.train_batch_part*4 + ifile]
                print(snli_train_file)
                print(mnli_train_file)
                print("")

                if snli_train_file:
                    batch_train_data.extend(pickle.load(
                        open(self.batch_dir + snli_train_file, 'rb')))
                batch_train_data.extend(pickle.load(
                    open(self.batch_dir + mnli_train_file, 'rb')))

            if len(batch_valid_data) == 0:
                batch_valid_data = pickle.load(
                    open(self.batch_dir + snli_test_file_list[0], 'rb'))
                batch_valid_data.extend(pickle.load(
                    open(self.batch_dir + mnli_test_file_list[0], 'rb')))
            random.shuffle(batch_train_data)

    def _init_batch_old(self):
        """
        Read dataset into memory
        """
        global batch_train_data, batch_valid_data

        if not self.valid:
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
            return int(len(batch_train_data)/10)

    def __getitem__(self, idx):
        """
        Returns batch.
        """
        global batch_train_data, batch_valid_data

        if torch.is_tensor(idx):
            idx = idx.tolist()

        sentence1 = torch.zeros((self.config['max_sent_len'], self.config['word_edim'],),
                                dtype=torch.float)
        sentence1_mask = torch.ones((self.config['max_sent_len'],),
                                    dtype=torch.bool)
        sentence2 = torch.zeros((self.config['max_sent_len'], self.config['word_edim'],),
                                dtype=torch.float)
        sentence2_mask = torch.ones((self.config['max_sent_len'],),
                                    dtype=torch.bool)
        # label = torch.zeros((3,), dtype=torch.float)
        label = torch.zeros((1,), dtype=torch.long)

        test_words = torch.zeros((self.config['max_sent_len'], self.config['word_edim'],),
                                  dtype=torch.float)
        test_labels = torch.zeros((self.config['max_sent_len'], 2), dtype=torch.float)

        batch_dataset = batch_valid_data if self.valid else batch_train_data

        idx = random.randint(0, len(batch_dataset)-1)

        sent1 = batch_dataset[idx]['sentences_emb'][0]
        sent2 = batch_dataset[idx]['sentences_emb'][1]
        nli_label = batch_dataset[idx]['label']

        sentence1[0:min(len(sent1), self.config['max_sent_len'])] =\
            torch.from_numpy(sent1[0:min(len(sent1), self.config['max_sent_len'])].astype(np.float32))
        sentence1_mask[0:min(len(sent1), self.config['max_sent_len'])] = torch.tensor(0.0)

        sentence2[0:min(len(sent2), self.config['max_sent_len'])] =\
            torch.from_numpy(sent2[0:min(len(sent2), self.config['max_sent_len'])].astype(np.float32))
        sentence2_mask[0:min(len(sent2), self.config['max_sent_len'])] = torch.tensor(0.0)

        # label[nli_label] = 1.0
        label = nli_label

        for test_idx in range(self.config['max_sent_len']):
            rnd = random.choice([False, True])
            if rnd:
                rnd_word = random.randint(0, min(len(sent2), self.config['max_sent_len'])-1)
                test_words[test_idx] = torch.from_numpy((sent2[rnd_word] + np.random.normal(scale=0.3, size=(self.config['word_edim']))).astype(np.float32))
            else:
                rnd_batch = random.randint(0, len(batch_dataset)-2)
                if rnd_batch == idx:
                    rnd_batch = len(batch_dataset)-1
                rnd_sent = batch_dataset[rnd_batch]['sentences_emb'][1]
                rnd_word = random.randint(0, len(rnd_sent)-1)
                test_words[test_idx] = torch.from_numpy((rnd_sent[rnd_word] + np.random.normal(scale=0.3, size=(self.config['word_edim']))).astype(np.float32))
            test_labels[test_idx][int(rnd)] = 1.0

        return sentence1, sentence1_mask, sentence2, sentence2_mask, label, test_words, test_labels

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
