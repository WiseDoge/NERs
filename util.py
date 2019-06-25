from typing import List, Dict, Optional, Tuple, Mapping
import torch
import os
import pickle


def load_dict(word2ix_path, tag2ix_path) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Load vocab and tag dict from file.

    The type of dict is python dict object.

    Returns:
        vocab dict, tag dict
    """
    with open(word2ix_path, "rb") as f:
        word_to_ix = pickle.load(f)
    with open(tag2ix_path, "rb") as f:
        tag_to_ix = pickle.load(f)
    return word_to_ix, tag_to_ix


def load_data(filename:str) -> List[List[str]]:
    """Load data from file

    This function only read the first and last column of input file.

    File format:
        sent1word1 [sent1word1 attr1] ... [sent1word1 attr_n] sent1tag1
        sent1word2 [sent1word1 attr1] ... [sent1word1 attr_n] sent1tag2
        sent1word3 [sent1word1 attr1] ... [sent1word1 attr_n] sent1tag2

        sent2word1 [sent1word1 attr1] ... [sent1word1 attr_n] sent2tag1
        sent2word2 [sent1word1 attr1] ... [sent1word1 attr_n] sent2tag2

        sent3word1 [sent1word1 attr1] ... [sent1word1 attr_n] sent3tag1
        sent3word2 [sent1word1 attr1] ... [sent1word1 attr_n] sent3tag2
        sent3word3 [sent1word1 attr1] ... [sent1word1 attr_n] sent3tag3
    """
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
                lst.append([line[0], line[-1]])
    return datas


def convert_tokens_to_ids(datas:List[List[str]], 
                          maxlen:int, 
                          word_to_ix:Mapping[str, int], 
                          tag_to_ix: Optional[Mapping[str, int]]=None) -> torch.LongTensor:
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
                dataset[i, j, 0] = word_to_ix[word] if word in word_to_ix else word_to_ix['[UNK]']
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


