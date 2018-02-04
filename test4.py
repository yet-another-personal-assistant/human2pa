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
        if max(res[0]) > 0.1:
            return self.lb.inverse_transform(res)[0]
        else:
            return 'unknown'

    def embed(self, sentence):
        return hashing_trick(sentence, HASH_SIZE, 'md5')

    def keras_eval(self, lines, labels):
        X = pad_sequences([self.embed(line) for line in lines],
                            padding='post', maxlen=self.max_length)
        y = self.lb.transform(labels)
        loss, accuracy = self.model.evaluate(X, y)
        print(loss, accuracy*100)


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

    tst_en = os.path.join(data_dir, "train.en")
    tst_pa = os.path.join(data_dir, "train.pa")
    cnt = 0
    wrong = 0
    with open(tst_en) as fen, open(tst_pa) as fpa:
        for en, pa in zip(fen.readlines(), fpa.readlines()):
            cls = translator.classify(en)
            act = pa.split(maxsplit=1)[0]
            cnt += 1
            if cls != act:
                wrong += 1
                print(act, cls, en.strip())
    print(cnt, wrong, (cnt-wrong)/cnt*100)


if __name__ == '__main__':
    main()
