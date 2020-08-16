import numpy as np
import os.path
import pickle
import tensorflow as tf
from typing import Optional
import random
random.seed(0)
import torch
from torch.utils.data import Dataset

batch_train_data = []
batch_valid_data = []

class AutoencoderBatch(Dataset):
    def __init__(self, config, valid=False):
        self.valid = valid
        self.config = config
        self.batch_dir = '../s2v_linker/train/_datasets_1s/'
        self._init_batch()

    def _init_batch(self):
        global batch_train_data, batch_valid_data

        if not self.valid:
            batch_train_data = []
            batch_valid_data = []
            file_list = []
            batch_files = os.listdir(self.batch_dir)
            for batch in batch_files:
                if 'wEmb' in batch:
                    file_list.append(batch)
            random.shuffle(file_list)
            for file in file_list[:2]:
                data = pickle.load(open(self.batch_dir + file, 'rb'))
                for title_idx, title in enumerate(data):
                    if title_idx == 0:
                        batch_valid_data.append(data[title])
                    else:
                        batch_train_data.append(data[title])

    def on_epoch_end(self):
        self._init_batch()

    def __len__(self):
        global batch_train_data, batch_valid_data

        if self.valid:
            return int(len(batch_valid_data)//2)
        else:
            return int(len(batch_train_data)//2)

    def __getitem__(self, idx):
        """
        Returns batch.
        """
        global batch_train_data, batch_valid_data

        if torch.is_tensor(idx):
            idx = idx.tolist()

        sentence = torch.zeros((2, self.config['max_sent_len'], self.config['word_edim'],),
                                dtype=torch.float)
        sentence_mask = torch.ones((2, self.config['max_sent_len'],),
                                    dtype=torch.bool)
        words = torch.zeros((2, self.config['max_sent_len'], self.config['word_edim'],),
                                  dtype=torch.float)
        words_labels = torch.zeros((2, self.config['max_sent_len'], 2), dtype=torch.float)


        batch_dataset = batch_valid_data if self.valid else batch_train_data

        for s_idx in range(2):
            idx = random.randint(0, len(batch_dataset)-1)

            sent = batch_dataset[idx]['first_sentence_emb'][0]

            sentence[s_idx][0:min(len(sent), self.config['max_sent_len'])] =\
                torch.from_numpy(sent[0:min(len(sent), self.config['max_sent_len'])].astype(np.float32))
            sentence_mask[s_idx][0:min(len(sent), self.config['max_sent_len'])] = torch.tensor(0.0)

            for test_idx in range(self.config['max_sent_len']):
                rnd = random.choice([False, True])
                if rnd:
                    rnd_word = random.randint(0, min(len(sent), self.config['max_sent_len'])-1)
                    words[s_idx][test_idx] = torch.from_numpy((sent[rnd_word] + np.random.normal(scale=0.3, size=(self.config['word_edim']))).astype(np.float32))
                else:
                    rnd_batch = random.randint(0, len(batch_dataset)-2)
                    if rnd_batch == idx:
                        rnd_batch = len(batch_dataset)-1
                    rnd_sent = batch_dataset[rnd_batch]['first_sentence_emb'][0]
                    rnd_word = random.randint(0, len(rnd_sent)-1)
                    words[s_idx][test_idx] = torch.from_numpy((rnd_sent[rnd_word] + np.random.normal(scale=0.3, size=(self.config['word_edim']))).astype(np.float32))
                words_labels[s_idx][test_idx][int(rnd)] = 1.0

        return sentence[0], sentence_mask[0], sentence[1], sentence_mask[1], words[0], words_labels[0], words[1], words_labels[1]

def test():
    batcher = AutoencoderBatch({
        'batch_size': 2,
        'max_sent_cnt': 6,
        'max_sent_len': 64,
        'word_edim': 1024
    })
    batch_x, batch_y = batcher.__getitem__(1)
    print(batch_x['sentences'][0])

# test()
