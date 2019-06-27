#!/usr/bin/env python3
import random
import sys

import keras
import numpy as np


# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
emb_size = 7
epochs = 1
verbose = 0


def load_doc(path):
    with open(path) as f:
        return f.read()


class TrainingData:

    def __init__(self, path):
        self.data = load_doc(path)
        chars = list(set(self.data))
        self.data_size = len(self.data)
        self.vocab_size = len(chars)
        print('data has %d characters, %d unique.' % (self.data_size, self.vocab_size))
        self.char_to_ix = { ch:i for i,ch in enumerate(chars) }
        self.ix_to_char = { i:ch for i,ch in enumerate(chars) }

        self.upper = [self.char_to_ix[c] for c in self.data if c.isupper()]


def sample(model, seed_ix, n, vocab_size):
    x = np.zeros((1, 1))
    x[0, 0] = seed_ix
    yield seed_ix
    for t in range(n):
        y = model.predict(x)[0]
        ix = np.random.choice(range(vocab_size), 1, p=(y/np.sum(y)).ravel())[0]
        x[0, 0] = ix
        yield ix


def main(path):
    data = TrainingData(path)

    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(data.vocab_size, emb_size))
    model.add(keras.layers.SimpleRNN(hidden_size))
    model.add(keras.layers.Dense(data.vocab_size, activation='softmax', name='output'))
    model.compile(optimizer='adagrad', loss='categorical_crossentropy')

    n = 0
    smooth_loss = -np.log(1.0/data.vocab_size)*seq_length # loss at iteration 0
    smoother = 0.001 * (len(data.data) // seq_length) * epochs
    X = np.ndarray((len(data.data)-1, 1), dtype=int)
    y = np.zeros((len(data.data)-1, data.vocab_size), dtype=int)
    for t, (i, o) in enumerate(zip(data.data, data.data[1:])):
        if t % 100 == 0:
            print(t, '/', len(data.data), end='\r')
            sys.stdout.flush()
        X[t, 0] = data.char_to_ix[i]
        y[t, data.char_to_ix[o]] = 1
    try:
        while True:
            model.reset_states()

            hist = model.fit(X, y, batch_size=seq_length, epochs=epochs, verbose=verbose)
            loss = hist.history['loss'][0]
            smooth_loss = smooth_loss * (1-smoother) + loss * smoother

            model.reset_states()
            print('iter %d, loss: %f' % (n, smooth_loss)) # print progress
            sample_ix = sample(model, random.choice(data.upper), 200, data.vocab_size)
            txt = ''.join(data.ix_to_char[ix] for ix in sample_ix)
            print('----\n', txt, '\n----')

            n += 1 # iteration counter
    except KeyboardInterrupt:
        np.save("embedding.npy", model.layers[0].get_weights())
        model.save('model.h5')
        with open('vocab.txt', 'w') as vocab:
            for i in range(data.vocab_size):
                print(data.ix_to_char[i], end='', file=vocab)


if __name__ == '__main__':
    main('data/input.txt')
