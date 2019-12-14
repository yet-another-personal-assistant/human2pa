#!/usr/bin/env python3
import argparse
import logging
import sys

from pathlib import Path

import human2pa.gen_data
import human2pa.infer
import human2pa.tlk
import human2pa.train


def pretrain_lm(path):
    tlk_corpus = human2pa.tlk.get_data(path=path)
    mappings = human2pa.train.make_vocab(tlk_corpus, path=path)
    flair_corpus = human2pa.train.make_lm_corpus(tlk_corpus, path=path)

    human2pa.train.train_lm_model(flair_corpus, mappings, path / 'tlk-forward', forward=True,
                                  batch_size=50, epochs=50)
    human2pa.train.train_lm_model(flair_corpus, mappings, path / 'tlk-backward', forward=False,
                                  batch_size=50, epochs=50)


def train_cls(path):
    cls_corpus_file = path / 'labels.txt'
    if not cls_corpus_file.exists():
        human2pa.gen_data.generate('ru_training', data_dir=path)

    cls_corpus = human2pa.train.make_cls_corpus(cls_corpus_file, path=path)

    embeddings = human2pa.infer.make_embeddings(path=path, prefix='tlk')
    human2pa.train.train_cls_model(cls_corpus, embeddings, path / 'model-cls')


def train_tagger(path):
    tag_corpus_file = path / 'tags.txt'
    if not tag_corpus_file.exists():
        human2pa.gen_data.generate('ru_training', data_dir=path)

    tag_corpus = human2pa.train.make_tag_corpus(tag_corpus_file, path=path)

    embeddings = human2pa.infer.make_embeddings(path=path, prefix='tlk')
    human2pa.train.train_tag_model(tag_corpus, embeddings, path / 'model-tag')


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="model")
    subparsers = parser.add_subparsers(dest='command')
    subparsers.add_parser('lm')
    subparsers.add_parser('cls')
    subparsers.add_parser('tag')
    args = parser.parse_args()

    path = Path(args.model)
    if args.command == 'lm':
        pretrain_lm(path)
    elif args.command == 'cls':
        train_cls(path)
    elif args.command == 'tag':
        train_tagger(path)

