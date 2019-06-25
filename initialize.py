from util import load_data
from config import *
import pickle
import os


def word_tag_to_dict(data, word_to_ix, tag_to_ix):
    for line in data:
        for word in line:
            if word[0] not in word_to_ix:
                word_to_ix[word[0]] = len(word_to_ix)
            if word[1] not in tag_to_ix:
                tag_to_ix[word[1]] = len(tag_to_ix)


def create_save_dict(datas, word_dict_path, tag_dict_path):
    word_to_ix, tag_to_ix = {'[PAD]': 0, '[UNK]': 1}, {'[PAD]': 0}
    for data in datas:
        word_tag_to_dict(data, word_to_ix, tag_to_ix)

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
    create_save_dict([train_data, dev_data, test_data], word_dict_path, tag_dict_path)

if __name__ == "__main__":
    init(TRAIN_FILE_NAME, DEV_FILE_NAME, TEST_FILE_NAME, 
         WORD_DICT_PATH, TAG_DICT_PATH)
