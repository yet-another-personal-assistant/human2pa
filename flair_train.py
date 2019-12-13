#!/usr/bin/env python3
import logging
import sys

from pathlib import Path

import human2pa.tlk
import human2pa.train


def main():
    path = Path('model')
    tlk_corpus = human2pa.tlk.get_data(path=path)
    mappings = human2pa.train.make_vocab(tlk_corpus, path=path)
    flair_corpus = human2pa.train.make_corpus(tlk_corpus, path=path)

    human2pa.train.train_model(flair_corpus, mappings, path / 'tlk-forward', forward=True)
    human2pa.train.train_model(flair_corpus, mappings, path / 'tlk-backward', forward=False)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    main()
