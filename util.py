from typing import List, Dict, Tuple, Mapping
import torch
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


def load_data(filename: str, load_tag=True):
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
    seqs = []
    tags = []
    seq = []
    tag = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if line == '\n':
                if len(seq) == 0:
                    continue
                seqs.append(seq)
                tags.append(tag)
                seq = []
                tag = []
            else:
                line = line.split()
                seq.append(line[0])
                tag.append(line[-1])
    if load_tag:
        return seqs, tags
    else:
        return seqs


def convert_tokens_to_ids(datas: List[List[str]],
                          maxlen: int,
                          token_to_ix: Mapping[str, int]) -> torch.LongTensor:
    token_ids = torch.ones(len(datas), maxlen, dtype=torch.long) * token_to_ix['[PAD]']
    for (i, line) in enumerate(datas):
        if len(line) > maxlen:
            line = line[:maxlen]
        for j in range(len(line)):
            if '[UNK]' in token_to_ix:
                token_ids[i, j] = token_to_ix.get(line[j], token_to_ix['[UNK]'])
            else:
                token_ids[i, j] = token_to_ix[line[j]]
    return token_ids
