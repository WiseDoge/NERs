import torch
import os
import pickle
from config import *


def load_dict():
    with open(os.path.join(OUTPUT_DIR, 'word_to_ix.dict'), "rb") as f:
        word_to_ix = pickle.load(f)
    with open(os.path.join(OUTPUT_DIR, 'tag_to_ix.dict'), "rb") as f:
        tag_to_ix = pickle.load(f)
    return word_to_ix, tag_to_ix


def load_data(filename):
    datas = []
    lst = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if line == '\n':
                if len(lst) == 0:
                    continue
                datas.append(lst)
                lst = []
            else:
                line = line.split()
                lst.append(line)
    return datas


def convert_tokens_to_ids(datas, maxlen, word_to_ix, tag_to_ix=None):
    if tag_to_ix:
        dataset = torch.zeros(len(datas), maxlen, 2)
    else:
        dataset = torch.zeros(len(datas), maxlen)
    for (i, line) in enumerate(datas):
        if len(line) > maxlen:
            line = line[:maxlen]
        for j in range(len(line)):
            word, pos = line[j]
            if tag_to_ix:
                dataset[i, j, 0] = word_to_ix[word]
                dataset[i, j, 1] = tag_to_ix[pos]
            else:
                dataset[i, j] = word_to_ix[word]
        for j in range(len(line), maxlen):
            if tag_to_ix:
                dataset[i, j, 0] = word_to_ix['[PAD]']
                dataset[i, j, 1] = tag_to_ix['[PAD]']
            else:
                dataset[i, j] = word_to_ix['[PAD]']
    return dataset.long()


if __name__ == "__main__":
    sent = ['Li', 'Youbin', 'starred', 'in', 'the', 'liang', 'jian']
    tags = ['I-PER', 'I-PER', 'O', 'O', 'O', 'I-TV', 'I-TV']
    print(extra_entity(sent, tags))
    sent = ['A', 'A', 'A', 'B', 'B', 'C']
    tags = ['I-PER', 'I-PER', 'B-PER', 'I-PER', 'O']
    print(extra_entity(sent, tags))
    tags = ['I-PER', 'I-PER', 'I-T', 'I-T', 'I-T', 'I-M']
    print(extra_entity(sent, tags))
