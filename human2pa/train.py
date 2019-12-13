import glob
import logging
import pickle

from collections import Counter
from pathlib import Path

import sklearn.model_selection

from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus


_LOGGER = logging.getLogger(__name__)


def make_vocab(corpus, path=Path('.')):
    char_dictionary: Dictionary = Dictionary()
    counter = Counter()

    with open(corpus, 'r', encoding='utf-8') as f:
        for line in f:
            counter.update(list(line))

    total_count = 0
    for letter, count in counter.most_common():
        total_count += count

    sum = 0
    idx = 0
    for letter, count in counter.most_common():
        sum += count
        percentile = (sum / total_count)
        char_dictionary.add_item(letter)
        idx += 1
        _LOGGER.info('%d\t%s\t%7d\t%7d\t%f' % (idx, letter, count, sum, percentile))

    _LOGGER.info("%s", char_dictionary.item2idx)

    result = path / 'flair_char_mappings.pickle'

    with open(result, 'wb') as f:
        mappings = {'idx2item': char_dictionary.idx2item,
                    'item2idx': char_dictionary.item2idx}
        pickle.dump(mappings, f)

    return result


def make_corpus(data, path=Path('.')):
    result = path / 'corpus'
    train_dir = result / 'train'
    train_dir.mkdir(parents=True, exist_ok=True)

    with open(data) as f:
        lines = f.readlines()

    train, tv = sklearn.model_selection.train_test_split(lines, test_size=0.1)
    test, valid = sklearn.model_selection.train_test_split(tv, test_size=0.5)

    _LOGGER.info("Train: %d lines", len(train))
    _LOGGER.info("Test: %d lines", len(test))
    _LOGGER.info("Validation: %d lines", len(valid))

    with open(train_dir / 'train.txt', 'w') as out:
        for text in train:
            print(text.strip(), file=out)
    with open(result / "test.txt", "w") as out:
        for text in test:
            print(text.strip(), file=out)
    with open(result / "valid.txt", "w") as out:
        for text in valid:
            print(text.strip(), file=out)

    return result


def train_model(corpus, mappings, model, forward=True, batch_size=50, epochs=50):
    dictionary: Dictionary = Dictionary.load_from_file(mappings)

    corpus = TextCorpus(corpus,
                        dictionary,
                        forward,
                        character_level=True)

    language_model = LanguageModel(dictionary,
                                   forward,
                                   hidden_size=128,
                                   nlayers=1)

    trainer = LanguageModelTrainer(language_model, corpus)

    trainer.train(model,
                  sequence_length=10,
                  mini_batch_size=batch_size,
                  max_epochs=epochs)
