#!/usr/bin/env python3
import argparse
import glob
import os
import sys

import keras
import numpy as np
import sklearn.model_selection

from char.data import text_to_training
from char.embedding import CharEmbed


CONTEXT = 10
EMBEDDING_DIM = 7
HIDDEN_DIM = 100


def build_model(ce):
    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(ce.vocab_size, EMBEDDING_DIM,
                                     input_length=CONTEXT,
                                     mask_zero=0))
    model.add(keras.layers.LSTM(HIDDEN_DIM, activation='relu'))
    model.add(keras.layers.Dense(ce.vocab_size, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    return model


def get_model(model_file=None, embeddings_file=None, ce=None):
    if model_file is not None and os.path.exists(model_file):
        model = keras.models.load_model(model_file)
    else:
        model = build_model(ce)

    if embeddings_file is not None and os.path.exists(embeddings_file):
        weights = np.load(embeddings_file)
        model.layers[0].set_weights(weights)

    return model


def texts_to_training_data(data, ce):
    count = sum(map(len, data)) + len(data)
    X = np.ndarray((count, CONTEXT), dtype=int)
    y = np.ndarray((count, ce.vocab_size), dtype=int)

    position = 0
    for text in data:
        for i, (ox, oy) in enumerate(text_to_training(text, CONTEXT, 0)):
            X[position+i, :] = [ce.char2idx[c] for c in ox]
            y[position+i, :] = ce.onehot(oy)
        position += i+1
    return X, y


class MyGenerator(keras.utils.Sequence):
    def __init__(self, texts, ce, batch_size):
        self._texts = texts
        self._ce = ce
        self._batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self._texts) / float(self._batch_size)))

    def __getitem__(self, i):
        begin = i * self._batch_size
        end = begin + self._batch_size
        return texts_to_training_data(self._texts[begin:end], self._ce)


class EpochEnd(keras.callbacks.Callback):
    def __init__(self, model, ce, test_data):
        self._model = model
        self._ce = ce
        self._test_data = test_data
        self._iter = 0

    def on_epoch_end(self, epoch, logs=None):
        self._iter += 1
        print(self._iter, "Evaluating... ", end='')
        sys.stdout.flush()
        loss, accuracy = self._model.evaluate(self._test_data, verbose=0)
        print('Accuracy: %f' % (accuracy*100))
        generate(None, None, self._model, self._ce, 5)


def train(args, opts):
    print("Training")

    training_data = opts
    if args.training_data is not None:
        with open(args.training_data) as training_list:
            for line in training_list:
                if not line.strip().startswith('#'):
                    training_data.append(line.strip())
    data = []
    for path_pattern in training_data:
        for path in glob.glob(os.path.expanduser(path_pattern)):
            with open(path) as i:
                data.extend(i.read().splitlines())

    if os.path.exists(args.vocabulary):
        with open(args.vocabulary) as vocab:
            ce = CharEmbed(vocab=vocab.read())
    else:
        all_chars = set()
        for f in data:
            all_chars |= set(f)
        ce = CharEmbed(data=all_chars)
        with open(args.vocabulary, "w") as v:
            v.write(''.join(ce.idx2char[1:]))

    model = get_model(args.model, args.embeddings, ce)

    data, data_test = sklearn.model_selection.train_test_split(data, test_size=0.1)
    test_gen = MyGenerator(data_test, ce, args.batch_size)

    for iter in range(args.times):
        data_t, data_v = sklearn.model_selection.train_test_split(data, test_size=0.1)
        train_gen = MyGenerator(data_t, ce, args.batch_size)
        valid_gen = MyGenerator(data_v, ce, args.batch_size)
        model.fit(train_gen, epochs=args.epochs, validation_data=valid_gen,
                  callbacks=[keras.callbacks.ModelCheckpoint(args.model),
                             EpochEnd(model, ce, test_gen)])
        np.save(args.embeddings, model.layers[0].get_weights())
        model.save(args.model)


def generate(args, opts, model=None, ce=None, count=None):
    if ce is None:
        with open(args.vocabulary) as vocab:
            ce = CharEmbed(vocab=vocab.read())
    if model is None:
        model = get_model(args.model, args.embeddings, ce)

    if count is None:
        count = args.count

    for _ in range(count):
        result = []
        X = np.zeros((1, CONTEXT))
        while True:
            y = model.predict(X)[0]
            l = np.random.choice(range(len(y)), p=y/sum(y))
            if l == 0:
                break
            result.append(ce.idx2char[l])
            X[0, :-1] = X[0, 1:]
            X[0, -1] = l
        print(''.join(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--prefix", "-p", default=None)
    parser.add_argument("--embeddings", "-e", default="embedding.npy")
    parser.add_argument("--model", "-m", default="model.h5")
    parser.add_argument("--vocabulary", "-v", default="vocab.txt")

    subparsers = parser.add_subparsers(dest='command')

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument("--training-data")
    train_parser.add_argument('-e', '--epochs', type=int, default=50)
    train_parser.add_argument('-t', '--times', type=int, default=1)
    train_parser.add_argument('-b', '--batch-size', type=int, default=1000)

    generate_parser = subparsers.add_parser('generate')
    generate_parser.add_argument('-c', '--count', type=int, default=10)

    args, opts = parser.parse_known_args()

    if args.prefix is not None:
        args.prefix = os.path.expanduser(args.prefix)
        if not os.path.exists(args.prefix):
            os.makedirs(args.prefix)
        args.embeddings = os.path.join(args.prefix, args.embeddings)
        args.model = os.path.join(args.prefix, args.model)
        args.vocabulary = os.path.join(args.prefix, args.vocabulary)

    {
        'train': train,
        'generate': generate,
    }[args.command](args, opts)
