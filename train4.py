#!/usr/bin/env python3
import os
import pickle

from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


class MyTranslator:

    def __init__(self):
        self.max_length = 32
        self.lb = LabelBinarizer()
        self.w2v = Word2Vec(iter=1)
        self.model = Sequential()

    def build_keras_model(self):
        pass

    def keras_train(self, lines, labels):
        self.w2v.build_vocab(lines)
        self.w2v.train(lines, total_examples=len(lines), epochs=1)
        embedding = self.w2v.wv.get_keras_embedding()
        self.model.add(Flatten(8))
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
        return one_hot(sentence, self.vocab_size)

    def save(self, file_name):
        with open(file_name+".cls", "w") as out:
            out.write(self.model.to_json())

        with open(file_name+".lb", "wb") as out:
            pickle.dump(self.lb, out)


def load_sentences(file_name):
    with open(file_name) as fen:
        return fen.readlines()

def load_labels(file_name):
    with open(file_name) as fpa:
        return [line.strip().split(maxsplit=2)[1] for line in fpa]

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
