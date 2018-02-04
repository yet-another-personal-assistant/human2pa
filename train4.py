#!/usr/bin/env python3
import os
import pickle

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import hashing_trick
from sklearn.preprocessing import LabelBinarizer

HASH_SIZE=1000


class MyTranslator:

    def __init__(self):
        self.max_length = 32
        self.lb = LabelBinarizer()
        self.model = Sequential()

    def build_keras_model(self):
        pass

    def keras_train(self, lines, labels):
        self.model.add(Embedding(HASH_SIZE, 8, input_length=self.max_length))
        self.model.add(Flatten())
        self.model.add(Dense(len(self.lb.classes_), activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

        X = pad_sequences([self.embed(line) for line in lines],
                            padding='post', maxlen=self.max_length)
        y = self.lb.transform(labels)
        self.model.fit(X, y, epochs=50, verbose=0)

    def keras_eval(self, lines, labels):
        X = pad_sequences([self.embed(line) for line in lines],
                            padding='post', maxlen=self.max_length)
        y = self.lb.transform(labels)
        loss, accuracy = self.model.evaluate(X, y)
        print(loss, accuracy*100)

    def embed(self, sentence):
        return hashing_trick(sentence, HASH_SIZE, 'md5')

    def save(self, file_name):
        self.model.save(file_name+".cls")
        with open(file_name+".lb", "wb") as out:
            pickle.dump(self.lb, out)


def load_sentences(file_name):
    with open(file_name) as fen:
        return fen.readlines()

def load_labels(file_name):
    with open(file_name) as fpa:
        return [line.strip().split(maxsplit=1)[0] for line in fpa]

def load_words(file_name):
    with open(file_name) as fen:
        return fen.readlines()


def main():
    this_dir = os.path.dirname(__file__)
    data_dir = os.path.join(this_dir, "gen_data")
    sentences = load_sentences(os.path.join(data_dir, "train.en"))
    labels = load_labels(os.path.join(data_dir, "train.pa"))

    translator = MyTranslator()
    translator.lb.fit(labels)

    words = load_words(os.path.join(data_dir, "vocab.en"))

    translator.vocab_size = len(words)

    translator.build_keras_model()
    translator.keras_train(sentences, labels)

    test_sentences = load_sentences(os.path.join(data_dir, "tst.en"))
    test_labels = load_labels(os.path.join(data_dir, "tst.pa"))
    translator.keras_eval(test_sentences, test_labels)

    test_sentences = load_sentences(os.path.join(data_dir, "dev.en"))
    test_labels = load_labels(os.path.join(data_dir, "dev.pa"))
    translator.keras_eval(test_sentences, test_labels)

    translator.save(os.path.join(data_dir, "trained"))


if __name__ == '__main__':
    main()
