#!/usr/bin/env python3
import argparse
import glob
import itertools
import os
import shutil
import sys

import keras
import sklearn.model_selection
import numpy as np

CONTEXT = 10
EMBEDDING_DIM = 7
HIDDEN_DIM = 75

with open("training-data") as files:
    source_paths = list(itertools.chain(
        *[glob.glob(os.path.expanduser(x.strip())) for x in files]))


class CharEmbed:

    def load_data(self, paths, daily_page_export=None):
        self.data = []
        for f in paths:
            with open(f) as i:
                self.data.append(i.read())
        text = ""
        ln = 0
        if daily_page_export is None:
            return
        with open(os.path.expanduser(daily_page_export)) as i:
            for line in i:
                ln += 1
                if "=========" in line:
                    self.data.append(text.strip())
                    text = ""
                    ln = 0
                    continue
                if ln < 6:
                    continue
                text += line

    def load_vocab(self):
        if os.path.exists("vocab.txt"):
            with open("vocab.txt") as v:
                self.vocab = list(v.read())
        else:
            v_set = set()
            for d in self.data:
                v_set.update(d)
            self.vocab = list(v_set)
            with open("vocab.txt", "w") as v:
                v.write(''.join(self.vocab))
        self.vocab.insert(0, '')

        self.vocab.append('<BR>')
        self.char2index = {c: i for i, c in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def onehot(self, char):
        result = np.zeros((self.vocab_size,))
        result[self.char2index[char]] = 1
        return result


def main(args):
    ce = CharEmbed()
    ce.load_data(source_paths, args.daily_page_export)
    ce.load_vocab()

    if args.debug:
        ce.data = ce.data[:2]

    all_data = list(itertools.chain(*ce.data))
    count = len(all_data) + len(ce.data)
    X = np.ndarray((count, CONTEXT), dtype=int)
    y = np.ndarray((count, ce.vocab_size), dtype=int)

    position = 0
    for text in ce.data:
        for i in range(CONTEXT-1):
            X[position, i] = 0
        X[position, CONTEXT-1] = ce.char2index['<BR>']

        for i, c in enumerate(text):
            y[position+i, :] = ce.onehot(c)
            X[position+i+1, :CONTEXT-1] = X[position+i, 1:]
            X[position+i+1, CONTEXT-1] = ce.char2index[c]
        y[position+i, :] = ce.onehot('<BR>')

        position += i+1

    if args.debug:
      for j in range(position):
        print("Position", j,
                ''.join(repr(ce.vocab[n] or '.') for n in X[j]),
                ce.vocab[(y[j]==1).tostring().find(b'\x01')])
      return

    if os.path.exists('model.h5'):
        model = keras.models.load_model('model.h5')
    else:
        model = keras.models.Sequential()
        model.add(keras.layers.Embedding(ce.vocab_size, EMBEDDING_DIM,
                                         input_length=CONTEXT,
                                         mask_zero=0))
        model.add(keras.layers.LSTM(HIDDEN_DIM))
        model.add(keras.layers.Dense(ce.vocab_size, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        if os.path.exists('embedding.npy'):
            weights = np.load('embedding.npy')
            model.layers[0].set_weights(weights)

    X, X_test, y, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

    for _ in range(args.times):
        Xt, Xv, yt, yv = sklearn.model_selection.train_test_split(X, y, test_size=0.25)
        model.fit(Xt, yt, epochs=args.epochs, validation_data=(Xv, yv))
        print("Evaluating... ", end='')
        sys.stdout.flush()
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print('Accuracy: %f' % (accuracy*100))
        np.save("embedding.npy", model.layers[0].get_weights())
        model.save('model.h5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('-t', '--times', type=int, default=1)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('--daily-page-export')
    args = parser.parse_args()
    main(args)
