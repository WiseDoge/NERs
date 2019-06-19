from util import load_data
from config import *
import pickle


def gen_dict(data, word_to_ix, tag_to_ix):
    for line in data:
        for word in line:
            if word[0] not in word_to_ix:
                word_to_ix[word[0]] = len(word_to_ix)
            if word[1] not in tag_to_ix:
                tag_to_ix[word[1]] = len(tag_to_ix)


def build_dict(datas):
    word_to_ix, tag_to_ix = {'[PAD]': 0, '[UNK]': 1}, {'[PAD]': 0}
    for data in datas:
        gen_dict(data, word_to_ix, tag_to_ix)
    with open(WORD_DICT_PATH, "wb") as f:
        pickle.dump(word_to_ix, f)
    with open(TAG_DICT_PATH, "wb") as f:
        pickle.dump(tag_to_ix, f)


def main():
    train_data = load_data(TRAIN_FILE_NAME)
    dev_data = load_data(DEV_FILE_NAME)
    test_data = load_data(TEST_FILE_NAME)
    build_dict([train_data, dev_data, test_data])


if __name__ == "__main__":
    main()
