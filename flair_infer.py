#!/usr/bin/env python3
import argparse

from pathlib import Path

from flair.data import Sentence

import human2pa.infer


def lm_embed(args, opts):
    path = Path(args.model)
    sentence = Sentence(' '.join(opts))
    embeddings = human2pa.infer.make_embeddings(path, 'tlk')
    embeddings.embed(sentence)
    print(sentence)
    for token in sentence:
        print(token, token.embedding)


def infer_class(args, opts):
    path = Path(args.model)
    sentence = Sentence(' '.join(opts))
    classifier = human2pa.infer.load_cls_model(path / 'model-cls')
    classifier.predict(sentence)
    print(sentence)


def infer_tags(args, opts):
    path = Path(args.model)
    sentence = Sentence(' '.join(opts))
    tagger = human2pa.infer.load_tag_model(path / 'model-tag')
    tagger.predict(sentence)
    print(sentence.to_tagged_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="model")
    subparsers = parser.add_subparsers(dest='command')
    subparsers.add_parser('lm')
    subparsers.add_parser('cls')
    subparsers.add_parser('tag')
    args, opts = parser.parse_known_args()

    if args.command == 'lm':
        lm_embed(args, opts)
    elif args.command == 'cls':
        infer_class(args, opts)
    elif args.command == 'tag':
        infer_tags(args, opts)
