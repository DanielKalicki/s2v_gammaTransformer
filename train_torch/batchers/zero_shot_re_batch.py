import os
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from flair.embeddings import RoBERTaEmbeddings
from flair.data import Sentence

class ZeroShotReBatch(Dataset):
    def __init__(self, config, valid=False):
        self.config = config
        self.valid = valid

        self.datasets_dir = "./datasets/relation_splits/"
        self.batch_dir = './train_torch/datasets/zero_shot_re/'
        self._create_batch_file()

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
        possible_answers = {}

        dataset_files = os.listdir(self.datasets_dir)
        for file in dataset_files:
            if ("train" in file) or ("test" in file):
                print(file)
                num_lines = sum(1 for line in open(self.datasets_dir + file, 'r'))
                with open(self.datasets_dir + file, "r") as f:
                    processed_dataset = []
                    part = 0
                    for line in tqdm(f, total=num_lines):
                        data = line.strip().split("\t")
                        if (len(data) >= 5) or True:
                            question_type = data[0]
                            question = data[1].replace("XXX", data[2])
                            sentence = data[3]
                            answers = data[4:]
                            if len(sentence.split(" ")) < 150:
                                # if not question_type in possible_answers:
                                #     possible_answers[question_type] = set()
                                # for answer in answers:
                                #     possible_answers[question_type].add(answer)
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
                                    pickle.dump(processed_dataset, open(self.batch_dir + file + '.' + str(part) + '.pickle', 'wb'))
                                    print(part)
                                    processed_dataset = []
                                    part += 1
                            else:
                                print(sentence)
                    if len(processed_dataset) > 0:
                        pickle.dump(processed_dataset, open(self.batch_dir + file + '.' + str(part) + '.pickle', 'wb'))
        # embed possible_answers

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
        pass
    def on_epoch_end(self):
        self._init_batch()
    def __len__(self):
        pass
    def __getitem__(self, idx):
        pass

def test():
    batcher = ZeroShotReBatch({
        'batch_size': 2,
        'max_sent_cnt': 6,
        'max_sent_len': 64,
        'word_edim': 1024
    })
    batch_x, batch_y = batcher.__getitem__(1)
    print(batch_x['sentences'][0])

test()