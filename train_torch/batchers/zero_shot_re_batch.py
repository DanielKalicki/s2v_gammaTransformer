import os
import pickle
import random
import torch
import math
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from flair.embeddings import RoBERTaEmbeddings
from flair.data import Sentence

batch_train_data = []
batch_valid_data = []

class ZeroShotReBatch(Dataset):
    def __init__(self, config, valid=False):
        self.config = config
        self.valid = valid

        self.datasets_dir = "./datasets/relation_splits/"
        self.batch_dir = './train_torch/datasets/zero_shot_re/'
        # self._create_batch_file()

        self.train_batch_part = -1
        self.valid_batch_part = -1
        self._init_batch()

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
        self._process_datasets()

    def _process_datasets(self):
        test_lines = set()
        train_lines = set()

        dataset_files = os.listdir(self.datasets_dir)
        for file in dataset_files:
            if ("train" in file) or ("test" in file):
                print(file)
                num_lines = sum(1 for line in open(self.datasets_dir + file, 'r'))
                with open(self.datasets_dir + file, "r") as f:
                    processed_dataset = []
                    for line in tqdm(f, total=num_lines):
                        if 'test' in file:
                            test_lines.add(line)
                        elif 'train' in file:
                            train_lines.add(line)
                            
        for file in ['train', 'test']:
            part = 0
            file = 'test'
            lines = train_lines if file == 'train' else test_lines
            processed_dataset = []
            for line in tqdm(lines, total=len(lines)):
                data = line.strip().split("\t")
                if (len(data) >= 5) or True:
                    question_type = data[0]
                    question = data[1].replace("XXX", data[2])
                    sentence = data[3]
                    answers = data[4:]
                    if len(sentence.split(" ")) < 100:
                        sents_emb, words = self._process_sentences(
                                    [question, sentence])
                        batch = {
                            'question_type': question_type,
                            'question': question,
                            'question_emb': sents_emb[0],
                            'question_words': words[0],
                            'sentence': sentence,
                            'sentence_emb': sents_emb[1],
                            'sentence_words': words[1],
                            'answers': answers
                        }
                        processed_dataset.append(batch)
                        if len(processed_dataset) >= 20000:
                            pickle.dump(processed_dataset, open(self.batch_dir + file + '_' + str(part) + '.pickle', 'wb'))
                            print(part)
                            processed_dataset = []
                            part += 1
                    else:
                        print(sentence)
            if len(processed_dataset) > 0:
                pickle.dump(processed_dataset, open(self.batch_dir + file + '_' + str(part) + '.pickle', 'wb'))

    def _process_sentences(self, sentences):
        sentences_emb = []
        words = []
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
                words.append([t.text for t in sent])
                sentences_emb.append(np.array(sentence_emb).astype(np.float16))
            except IndexError:
                print('IndexError')
                print(sentence)
                sentence_emb = [np.array(t.embedding).astype(np.float16)
                                for t in sent]
                sentences_emb.append(np.array(sentence_emb).astype(np.float16))
        sentences_emb_short = sentences_emb
        return sentences_emb_short, words

    def _init_batch(self):
        """
        Read dataset into memory
        """
        global batch_train_data, batch_valid_data

        if not self.valid:
            train_files_list = []
            test_files_list = []
            batch_files = os.listdir(self.batch_dir)
            for batch in batch_files:
                if 'train' in batch:
                    train_files_list.append(batch)
                elif 'test' in batch:
                    test_files_list.append(batch)

            print("zero shot re batcher")
            batch_train_data = []
            for _ in range(2):
                self.train_batch_part += 1
                if self.train_batch_part >= len(train_files_list):
                    self.train_batch_part = 0

                # TODO remove long sentences?

                batch_train_data.extend(pickle.load(
                    open(self.batch_dir + train_files_list[self.train_batch_part], 'rb')))
                print(train_files_list[self.train_batch_part])

            if len(batch_valid_data) == 0:
                for self.valid_batch_part in range(0, len(test_files_list)-1):
                    batch_valid_data.extend(pickle.load(
                        open(self.batch_dir + test_files_list[self.valid_batch_part], 'rb')))
                    print(test_files_list[self.valid_batch_part])
            random.shuffle(batch_train_data)
            random.shuffle(batch_valid_data)

    def on_epoch_end(self):
        self._init_batch()

    def __len__(self):
        global batch_train_data, batch_valid_data
        if self.valid:
            return int(len(batch_valid_data)//10)
        else:
            return int(len(batch_train_data)*.75)

    def __getitem__(self, idx):
        """
        Returns batch.
        """
        global batch_train_data, batch_valid_data

        if torch.is_tensor(idx):
            idx = idx.tolist()

        sentence1 = torch.zeros((self.config['max_sent_len'], self.config['word_edim'],), dtype=torch.float)
        sentence1_mask = torch.ones((self.config['max_sent_len'],), dtype=torch.bool)

        sentence2 = torch.zeros((self.config['max_sent_len'], self.config['word_edim'],), dtype=torch.float)
        sentence2_mask = torch.ones((self.config['max_sent_len'],), dtype=torch.bool)

        label = torch.zeros((1,), dtype=torch.long)

        batch_dataset = batch_valid_data if self.valid else batch_train_data

        sent1 = batch_dataset[idx]['question_emb']
        sent2 = batch_dataset[idx]['sentence_emb']

        sentence1[0:min(len(sent1), self.config['max_sent_len'])] =\
            torch.from_numpy(sent1[0:min(len(sent1), self.config['max_sent_len'])].astype(np.float32))
        sentence1_mask[0:min(len(sent1), self.config['max_sent_len'])] = torch.tensor(0.0)
        # sentence1_mask[0:min(int(math.ceil(len(sent1)/2)), self.config['max_sent_len'])] = torch.tensor(0.0)

        sentence2[0:min(len(sent2), self.config['max_sent_len'])] =\
            torch.from_numpy(sent2[0:min(len(sent2), self.config['max_sent_len'])].astype(np.float32))
        sentence2_mask[0:min(len(sent2), self.config['max_sent_len'])] = torch.tensor(0.0)
        # sentence2_mask[0:min(int(math.ceil(len(sent2)/2)), self.config['max_sent_len'])] = torch.tensor(0.0)

        # if not self.valid:
        #     sentence1 = sentence1 + torch.mean(sentence1)*torch.randn_like(sentence1)*0.1
        #     sentence2 = sentence2 + torch.mean(sentence2)*torch.randn_like(sentence2)*0.1

        label = 1 if len(batch_dataset[idx]['answers']) > 0 else 0

        return sentence1, sentence1_mask, sentence2, sentence2_mask, label

def test():
    batcher = ZeroShotReBatch({
        'batch_size': 2,
        'max_sent_cnt': 6,
        'max_sent_len': 64,
        'word_edim': 1024
    })
    batch_x, batch_y = batcher.__getitem__(1)
    print(batch_x['sentences'][0])

# test()