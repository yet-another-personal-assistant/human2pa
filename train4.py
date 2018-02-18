#!/usr/bin/env python3
import logging
import os
import time

import numpy as np

from gensim.models import Word2Vec
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import hashing_trick
from seq2seq.models import SimpleSeq2Seq
from sklearn.feature_extraction.text import TfidfVectorizer

from translator import Translator


def load_sentences(file_name):
    with open(file_name) as fen:
        return [l.strip() for l in fen.readlines()]


def load_labels(file_name):
    with open(file_name) as fpa:
        return [line.strip().split(maxsplit=1)[0] for line in fpa]


def test_tagger_train(sentences, labels, tags):
    count = int(len(sentences)/10)
    label_count = len(set(labels))
    translator2 = Translator(label_count)
    translator2.lb.fit(labels)
    translator2.train_tagger(sentences[:count], tags[:count])


def load_vocab(lines):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(lines)
    return vectorizer.vocabulary_


TOKEN_REPRESENTATION_SIZE = 256
VOCAB_MAX_SIZE = 20000
TOKEN_MIN_FREQUENCY = 1
INPUT_SEQUENCE_LENGTH = 32
HIDDEN_LAYER_DIMENSION = 512
ANSWER_MAX_TOKEN_LENGTH = 32
TRAIN_BATCH_SIZE = 50

def make_tagger_dataset(tags):
    tokenized_en_lines = [s.split() + ['$$$'] for s in tags]
    en_vocab = set()
    for s in tokenized_en_lines:
        en_vocab.update(s)
    en_vocab.add('###')
    assert(len(en_vocab) < VOCAB_MAX_SIZE) # TODO: actually discard stuff
    index_to_token = dict(enumerate(en_vocab))

    return tokenized_en_lines, index_to_token


def make_nn_model(output_dim):
    model = Sequential()
    model.add(Embedding(256, TOKEN_REPRESENTATION_SIZE,
                        input_length=INPUT_SEQUENCE_LENGTH))
    model.add(SimpleSeq2Seq(input_dim=TOKEN_REPRESENTATION_SIZE,
                            input_length=INPUT_SEQUENCE_LENGTH,
                            hidden_dim=HIDDEN_LAYER_DIMENSION,
                            output_dim=output_dim,
                            output_length=ANSWER_MAX_TOKEN_LENGTH,
                            depth=1))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


def _embed(sentence):
    return hashing_trick(sentence, 256, 'md5')
    

def make_training_data(en, tg, index2token):
    token2index = dict((v, k) for k, v in index2token.items())
    voc_size = len(token2index)
    X = pad_sequences([_embed(line) for line in en],
                      padding='post', maxlen=INPUT_SEQUENCE_LENGTH)
    Y = np.zeros((len(en), ANSWER_MAX_TOKEN_LENGTH, voc_size), dtype=np.bool)
    for i, s in enumerate(tg):
        for ti, t in enumerate(s):
            Y[i, ti, token2index[t]] = 1
    return X, Y


def train_nn_model(nn_model, en_lines, tagger_tg_dataset):
    tg_for_nn, index2token = tagger_tg_dataset

    X, Y = make_training_data(en_lines, tg_for_nn, index2token)
    nn_model.fit(X, Y, batch_size=TRAIN_BATCH_SIZE, epochs=10, verbose=1)


def main():
    this_dir = os.path.dirname(__file__)
    data_dir = os.path.join(this_dir, "gen_data")
    sentences = load_sentences(os.path.join(data_dir, "train.en"))
    labels = load_labels(os.path.join(data_dir, "train.pa"))
    tags = load_sentences(os.path.join(data_dir, "train.tg"))
    label_count = len(set(labels))

    translator = Translator(label_count)
    translator.lb.fit(labels)

    tagger_tg_dataset = make_tagger_dataset(tags)
    tagger_nn_model = make_nn_model(len(tagger_tg_dataset[1]))

    train_nn_model(tagger_nn_model, sentences, tagger_tg_dataset)

    return

    translator.train_classifier(sentences, labels,
                                validation=(v_sentences, v_labels))

    test_sentences = load_sentences(os.path.join(data_dir, "tst.en"))
    test_labels = load_labels(os.path.join(data_dir, "tst.pa"))
    translator.classifier_eval(test_sentences, test_labels)

    test_sentences = load_sentences(os.path.join(data_dir, "dev.en"))
    test_labels = load_labels(os.path.join(data_dir, "dev.pa"))
    translator.classifier_eval(test_sentences, test_labels)

    translator.save(os.path.join(data_dir, "trained"))


if __name__ == '__main__':
    main()
