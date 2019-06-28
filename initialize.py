from typing import List, Mapping, Tuple
from util import load_data
import pickle
import os


def token_to_dict(lines: List[List[str]], token_dict_ref: Mapping[str, int]) -> None:
    for line in lines:
        for token in line:
            if token not in token_dict_ref:
                token_dict_ref[token] = len(token_dict_ref)


def create_save_dict(datas: List[Tuple[List[List[str]], List[List[str]]]],
                     word_dict_path: str, tag_dict_path: str) -> None:
    word_to_ix, tag_to_ix = {'[PAD]': 0, '[UNK]': 1}, {'[PAD]': 0}
    for seqs, tags in datas:
        token_to_dict(seqs, word_to_ix)
        token_to_dict(tags, tag_to_ix)

    for filename in [word_dict_path, tag_dict_path]:
        dirname, _ = os.path.split(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    with open(word_dict_path, "wb") as f:
        pickle.dump(word_to_ix, f)
    with open(tag_dict_path, "wb") as f:
        pickle.dump(tag_to_ix, f)


def init(train, dev, test, word_dict_path, tag_dict_path):
    train_data = load_data(train)
    dev_data = load_data(dev)
    test_data = load_data(test)
    create_save_dict([train_data, dev_data, test_data],
                     word_dict_path, tag_dict_path)
