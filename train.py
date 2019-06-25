from util import *
from tagger import *

import os
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def do_train(train_filename, ouput_dir, word_dict_path, tag_dict_path, max_seq_len, embed_dim, hidden_dim, lr, batch_size, epochs, print_step, device):
    logger.info(f"***** Loading Training Data *****")
    train_data = load_data(train_filename)

    logger.info(f"***** Loading Dict *****")
    word_to_ix, tag_to_ix = load_dict(word_dict_path, tag_dict_path)

    logger.info(f"***** Generating Training Data *****")
    train_dataset = convert_tokens_to_ids(
        train_data, max_seq_len, word_to_ix, tag_to_ix)

    tag_dim = len(tag_to_ix)

    logger.info(f"***** Initializing Model *****")
    params = [len(word_to_ix), embed_dim, hidden_dim, tag_dim,
              lr, batch_size, epochs, device, tag_to_ix['[PAD]'], print_step]
    mask_model_params = [*params[:-1], word_to_ix['[PAD]'], params[-1]]
    taggers = [
        LRTagger(*params[:2], *params[3:]),
        HMMTagger(len(word_to_ix), tag_dim, tag_to_ix['[PAD]']),
        CNNTagger(*params),
        BiLSTMTagger(*params),
        BiLSTMCRFTagger(*mask_model_params),
        BiLSTMAttTagger(*mask_model_params),
        BiLSTMCNNTagger(*params),
        CNNBiLSTMTagger(*params),
        CNNBiLSTMAttTagger(*mask_model_params)
    ]

    for tagger in taggers:
        taggername = type(tagger).__name__

        logger.info(f"***** Training {taggername} *****")
        tagger.fit(train_dataset[..., 0], train_dataset[..., 1])

        logger.info(f"***** Saving {taggername} *****")
        tagger.save(os.path.join(ouput_dir, f'{taggername}_model.pt'))


