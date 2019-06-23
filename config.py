import os

BATCH_SIZE = 128
EPOCHS = 30
MAX_LEN = 180
HIDDEN_DIM = 256
WORD_EMBEDDING_DIM = 128
OUTPUT_DIR = 'output'
DATA_DIR = 'data'
EVAL_LOG_DIR = 'evallog'
LEARNING_RATE = 0.001
PRINT_STEP = 20

TRAIN_FILE_NAME = os.path.join(DATA_DIR, 'ResumeNER', 'train.char.bmes')
DEV_FILE_NAME = os.path.join(DATA_DIR, 'ResumeNER', 'dev.char.bmes')
TEST_FILE_NAME = os.path.join(DATA_DIR, 'ResumeNER', 'test.char.bmes')

WORD_DICT_PATH = os.path.join(OUTPUT_DIR, 'word_to_ix.dict')
TAG_DICT_PATH = os.path.join(OUTPUT_DIR, 'tag_to_ix.dict')
