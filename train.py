from util import *
from config import *
from tagger import *

import os
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info(f"***** Loading Training Data *****")
    train_data = load_data(TRAIN_FILE_NAME)

    logger.info(f"***** Loading Dict *****")
    word_to_ix, tag_to_ix = load_dict()

    logger.info(f"***** Generating Training Data *****")
    train_dataset = convert_tokens_to_ids(
        train_data, MAX_LEN, word_to_ix, tag_to_ix)

    tag_dim = len(tag_to_ix)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")

    logger.info(f"***** Initializing Model *****")
    params = [len(word_to_ix), WORD_EMBEDDING_DIM, HIDDEN_DIM, tag_dim,
              LEARNING_RATE, BATCH_SIZE, EPOCHS, device, tag_to_ix['[PAD]'], PRINT_STEP]
    taggers = [
        LRTagger(*params[:2], *params[3:]),
        HMMTagger(len(word_to_ix), tag_dim, tag_to_ix['[PAD]']),
        CNNTagger(*params),
        BiLSTMTagger(*params),
        BiLSTMCRFTagger(*params[:-1], word_to_ix['[PAD]'], params[-1]),
        BiLSTMAttTagger(*params[:-1], word_to_ix['[PAD]'], params[-1]),
        BiLSTMCNNTagger(*params),
        CNNBiLSTMTagger(*params),
        CNNBiLSTMAttTagger(*params[:-1], word_to_ix['[PAD]'], params[-1])
    ]

    for tagger in taggers:
        taggername = type(tagger).__name__

        logger.info(f"***** Training {taggername} *****")
        tagger.fit(train_dataset[..., 0], train_dataset[..., 1])

        logger.info(f"***** Saving {taggername} *****")
        tagger.save(os.path.join(OUTPUT_DIR, f'{taggername}_model.pt'))


if __name__ == "__main__":
    main()
