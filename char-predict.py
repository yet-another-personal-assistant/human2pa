#!/usr/bin/env python3
import argparse
import sys

import keras
import numpy as np

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())


def main(args):
    model = keras.models.load_model('model.h5')

    layer = model.layers[0]
    context = layer.input_length
    vocab_size = layer.input_dim

    with open("vocab.txt") as v:
        vocab = list(v.read())
    vocab.insert(0, '')
    if len(vocab) < vocab_size:
        vocab.extend(['<BR>'] * (vocab_size - len(vocab)))
    char2index = {c: i for i, c in enumerate(vocab)}

    starting_string = ' '.join(args.starting_string)

    if len(starting_string) < context-1:
        data = [0] * (context - len(starting_string)-1) + [char2index['<BR>']] + [char2index[c] for c in starting_string]
    else:
        data = [char2index['<BR>']] + [char2index[c] for c in starting_string[-context+1:]]
    if starting_string:
        print('['+starting_string+']', end='')
    x = np.ndarray((1, context))

    stop = args.stop
    if stop == r'\n':
        stop = '\n'
    count = 0
    while True:
        x[0] = data[:context]
        y = model.predict(x)[0]
        data = data[1:]
        if len(data) > context:
            print(vocab[data[context]], end='')
            if len(data) == context + 1:
                print(']', end='')
            continue

        l = np.random.choice(vocab, 1, p=y/sum(y))[0]
        if l == '<BR>':
            break
        if args.stop:
            if l == stop:
                count += 1
                if count == args.count:
                    break
        else:
            count += 1
            if count == args.length:
                break
        print(l, end='')
        sys.stdout.flush()
        data += [char2index[l]]
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--length', type=int, default=20)
    parser.add_argument('-s', '--stop')
    parser.add_argument('-c', '--count', type=int, default=1)
    parser.add_argument('starting_string', nargs='*')
    args = parser.parse_args()
    main(args)
