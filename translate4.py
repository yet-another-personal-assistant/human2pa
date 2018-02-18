#!/usr/bin/env python3
import os
import pickle
import sys

from keras.models import load_model

from translator import Translator


def main():
    this_dir = os.path.dirname(__file__)
    data_dir = os.path.join(this_dir, "gen_data")

    classifier = load_model(os.path.join(data_dir, "trained.cls"))
    tagger = load_model(os.path.join(data_dir, "trained.tagger"))
    with open(os.path.join(data_dir, "trained.lb"), 'rb') as labels_file:
        lb = pickle.load(labels_file)

    translator = Translator(lb=lb, cls=classifier, tagger=tagger)

    line = ' '.join(sys.argv)
    print(translator.classify2(line))
    print(translator.tag(line))


if __name__ == '__main__':
    main()
