#!/usr/bin/env python3
import keras
import numpy as np


# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1


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


def sample(model, seed_ix, n, vocab_size):
    x = np.zeros((1, 1, vocab_size))
    x[0, 0, seed_ix] = 1
    ixes = []
    for t in range(n):
        y = model.predict(x)[0]
        ix = np.random.choice(range(vocab_size), 1, p=(y/np.sum(y)).ravel())[0]
        x = np.zeros((1, 1, vocab_size))
        x[0, 0, ix] = 1
        ixes.append(ix)
    return ixes


def main(path):
    data = TrainingData(path)

    inp = keras.layers.Input(shape=(None, data.vocab_size))
    rnn = keras.layers.SimpleRNN(hidden_size)(inp)
    out = keras.layers.Dense(data.vocab_size, activation='softmax', name='output')(rnn)
    model = keras.models.Model(inp, out)
    model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['acc'])

    n, p = 0, 0
    smooth_loss = -np.log(1.0/data.vocab_size)*seq_length # loss at iteration 0
    while True:
        # prepare inputs (we're sweeping from left to right in steps seq_length long)
        if p+seq_length+1 >= len(data.data) or n == 0:
            model.reset_states()
            p = 0 # go from start of data
        inputs = [data.char_to_ix[ch] for ch in data.data[p:p+seq_length]]
        targets = [data.char_to_ix[ch] for ch in data.data[p+1:p+seq_length+1]]

        # sample from the model now and then
        if n % 100 == 0:
            sample_ix = sample(model, inputs[0], 200, data.vocab_size)
            txt = ''.join(data.ix_to_char[ix] for ix in sample_ix)
            print('----\n %s \n----' % (txt, ))

        X = np.ndarray((seq_length, 1, data.vocab_size), dtype=int)
        y = np.ndarray((seq_length, data.vocab_size), dtype=int)
        for t in range(seq_length):
            X[t, 0, :] = np.zeros(data.vocab_size)
            X[t, 0, inputs[t]] = 1
            y[t, :] = np.zeros(data.vocab_size)
            y[t, targets[t]] = 1

        # forward seq_length characters through the net and fetch gradient
        hist = model.fit(X, y, epochs=1, batch_size=seq_length, verbose=0)
        loss = hist.history['loss'][0]
        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        if n % 100 == 0:
            print('iter %d, loss: %f' % (n, smooth_loss)) # print progress

        p += seq_length # move data pointer
        n += 1 # iteration counter

if __name__ == '__main__':
    main('data/input.txt')
