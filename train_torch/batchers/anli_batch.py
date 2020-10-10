import numpy as np
import os.path
import pickle
import tensorflow as tf
from typing import Optional
import random
random.seed(0)
import torch
from torch.utils.data import Dataset
import json
import re
import math
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

        self.batch_dir = './train_torch/datasets/'
        self.labels = ['contradiction', 'neutral', 'entailment']

        self.word_list = self._load_word_list()
        self.word_dict = {}
        for widx, word in enumerate(self.word_list):
            self.word_dict[word] = widx
        print(len(self.word_list))

        # self.train_batch_part = -1
        self.snli_train_batch_part = -1
        self.mnli_train_batch_part = -1
        self._init_batch()

    def _process_word(self, word):
        word = word.lower()
        word = word.replace("!", "")
        word = word.replace("\\", "")
        word = word.replace(",", "")
        word = word.replace(".", "")
        word = word.replace("?", "")
        word = word.replace('"', "")
        word = word.replace("(", "")
        word = word.replace(")", "")
        return word

    def _load_word_list(self):
        words_file_exists = False
        words = {}
        batch_files = os.listdir(self.batch_dir)
        for file in batch_files:
            if 'words.json' in file:
                words_file_exists = True
                with open(self.batch_dir+file, 'r') as f:
                    words_list = json.load(f)
                    return words_list
        if not words_file_exists:
            for file in batch_files:
                if 'pickle' in file:
                    print(file)
                    data = pickle.load(open(self.batch_dir + file, 'rb'))
                    for batch in data:
                        for sent in batch['sentences_words']:
                            for word in sent:
                                word_ = self._process_word(word)
                                if not word_ in words:
                                    words[word_] = 0
                                words[word_] += 1
            words_list = []
            for word in words:
                if words[word] > 100:
                    words_list.append(word)
            with open(self.batch_dir+'words.json', 'w') as f:
                json.dump(words_list, f, indent=1)
                print(len(words_list))
                exit(0)
        return words_list

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
                elif ('train' in batch) and ('multinli' in batch):
                    mnli_train_files_list.append(batch)
                elif ('mismatched' in batch) and ('multinli' in batch):
                    mnli_test_file_list.append(batch)

            self.snli_train_batch_part += 1
            if self.snli_train_batch_part >= len(snli_train_files_list):
                self.snli_train_batch_part = 0
            self.mnli_train_batch_part += 1
            if self.mnli_train_batch_part >= len(mnli_train_files_list):
                self.mnli_train_batch_part = 0

            # self.snli_train_batch_part += 1
            # if self.train_batch_part >= len(snli_train_files_list)//1:
            #     self.train_batch_part = 0
            # print(self.train_batch_part)

            # snli_train_files_list.sort()
            # mnli_train_files_list.sort()
            # random.shuffle(snli_train_files_list)
            # random.shuffle(mnli_train_files_list)
            batch_train_data = []

            print(snli_train_files_list[self.snli_train_batch_part])
            print(mnli_train_files_list[self.mnli_train_batch_part])

            batch_train_data.extend(pickle.load(
                open(self.batch_dir + snli_train_files_list[self.snli_train_batch_part], 'rb')))
            batch_train_data.extend(pickle.load(
                open(self.batch_dir + mnli_train_files_list[self.mnli_train_batch_part], 'rb')))

            if len(batch_valid_data) == 0:
                batch_valid_data = pickle.load(
                    open(self.batch_dir + snli_test_file_list[0], 'rb'))
                batch_valid_data.extend(pickle.load(
                    open(self.batch_dir + mnli_test_file_list[0], 'rb')))
            random.shuffle(batch_train_data)
            random.shuffle(batch_valid_data)

    def on_epoch_end(self):
        self._init_batch()

    def __len__(self):
        global batch_train_data, batch_valid_data

        if self.valid:
            return len(batch_valid_data)
        else:
            # return int(len(batch_train_data)/1)
            return int(len(batch_train_data))

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
        label = torch.zeros((1,), dtype=torch.long)
        sent2_words = torch.zeros((len(self.word_list), 2), dtype=torch.float)

        batch_dataset = batch_valid_data if self.valid else batch_train_data

        # if not self.valid:
            # idx = random.randint(0, len(batch_dataset)-1)

        sent1 = batch_dataset[idx]['sentences_emb'][0]
        sent2 = batch_dataset[idx]['sentences_emb'][1]
        nli_label = batch_dataset[idx]['label']

        sentence1[0:min(len(sent1), self.config['max_sent_len'])] =\
            torch.from_numpy(sent1[0:min(len(sent1), self.config['max_sent_len'])].astype(np.float32))
        sentence1_mask[0:min(len(sent1), self.config['max_sent_len'])] = torch.tensor(0.0)
        # sentence1_mask[0:min(int(math.ceil(len(sent1)/2)), self.config['max_sent_len'])] = torch.tensor(0.0)

        sentence2[0:min(len(sent2), self.config['max_sent_len'])] =\
            torch.from_numpy(sent2[0:min(len(sent2), self.config['max_sent_len'])].astype(np.float32))
        sentence2_mask[0:min(len(sent2), self.config['max_sent_len'])] = torch.tensor(0.0)
        # sentence2_mask[0:min(int(math.ceil(len(sent2)/2)), self.config['max_sent_len'])] = torch.tensor(0.0)

        # for word_idx in range(len(self.word_list)):
        #     sent2_words[word_idx][0] = True
        # for word in batch_dataset[idx]['sentences_words'][1]:
        #     word_ = self._process_word(word)
        #     if word_ in self.word_dict:
        #         sent2_words[self.word_dict[word_]][1] = True

        # if not self.valid:
        #     word_to_drop = random.randint(0, min(len(sent1), self.config['max_sent_len'])-1)
        #     sentence1[word_to_drop] = torch.zeros((self.config['word_edim'],), dtype=torch.float)
        #     word_to_drop = random.randint(0, min(len(sent2), self.config['max_sent_len'])-1)
        #     sentence2[word_to_drop] = torch.zeros((self.config['word_edim'],), dtype=torch.float)

        if not self.valid:
            sentence1 = sentence1 + torch.mean(sentence1)*torch.randn_like(sentence1)*0.1
            sentence2 = sentence2 + torch.mean(sentence2)*torch.randn_like(sentence2)*0.1

        label = nli_label

        return sentence1, sentence1_mask, sentence2, sentence2_mask, label, sent2_words

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
