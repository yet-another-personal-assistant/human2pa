#!/usr/bin/env python3
import os
import pickle
import sys

from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from sklearn.preprocessing import LabelBinarizer 


class MyTranslator:

    def __init__(self, lb=None, model=None):
        self.max_length = 32
        self.lb = lb or LabelBinarizer()
        self.model = model

    def classify(self, line):
        X = pad_sequences([self.embed(line)],
                            padding='post', maxlen=self.max_length)
        res = self.model.predict(X)
        print(res)
        return self.lb.inverse_transform(res)

    def embed(self, sentence):
        return one_hot(sentence, self.vocab_size)


def main():
    this_dir = os.path.dirname(__file__)
    data_dir = os.path.join(this_dir, "gen_data")

    with open(os.path.join(data_dir, "trained.cls")) as model_file:
        model = model_from_json(model_file.read())
    with open(os.path.join(data_dir, "trained.lb"), 'rb') as labels_file:
        lb = pickle.load(labels_file)
    with open(os.path.join(data_dir, "vocab.en")) as vocab_file:
        vocab = vocab_file.readlines()

    translator = MyTranslator(lb, model)
    translator.vocab_size = len(vocab)

    print(translator.classify(' '.join(sys.argv)))


if __name__ == '__main__':
    main()
