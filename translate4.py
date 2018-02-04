#!/usr/bin/env python3
import os
import pickle
import sys

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import hashing_trick
from sklearn.preprocessing import LabelBinarizer

HASH_SIZE=1000


class MyTranslator:

    def __init__(self, lb=None, model=None):
        self.max_length = 32
        self.lb = lb or LabelBinarizer()
        self.model = model

    def classify(self, line):
        X = pad_sequences([self.embed(line)],
                          padding='post', maxlen=self.max_length)
        res = self.model.predict(X)
        print(res[0])
        m = max(res[0])
        c = self.lb.inverse_transform(res)[0]
        if m > 0.05:
            return c
        elif m > 0.02:
            return 'probably ' + c
        else:
            return 'unknown' 

    def embed(self, sentence):
        return hashing_trick(sentence, HASH_SIZE, 'md5')


def main():
    this_dir = os.path.dirname(__file__)
    data_dir = os.path.join(this_dir, "gen_data")

    model = load_model(os.path.join(data_dir, "trained.cls"))
    with open(os.path.join(data_dir, "trained.lb"), 'rb') as labels_file:
        lb = pickle.load(labels_file)
    with open(os.path.join(data_dir, "vocab.en")) as vocab_file:
        vocab = vocab_file.readlines()

    translator = MyTranslator(lb, model)
    translator.vocab_size = len(vocab)

    print(translator.classify(' '.join(sys.argv)))


if __name__ == '__main__':
    main()
