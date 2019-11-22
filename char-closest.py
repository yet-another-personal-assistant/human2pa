#!/usr/bin/env python3

import numpy as np
import scipy.spatial


def nearest(weights, vector, count):
    tree = scipy.spatial.cKDTree(weights)
    _, i = tree.query(vector, count+1)
    return i[1:]


def main():
    emb_weights = np.load('embedding.npy')[0]
    vocab_size = emb_weights.shape[0]

    with open("vocab.txt") as v:
        vocab = list(v.read())
    if len(vocab) < vocab_size:
        vocab.extend(['<BR>'] * (vocab_size - len(vocab)))
    char2index = {c: i for i, c in enumerate(vocab)}

    emb2char = {}
    for char, emb in zip(vocab, emb_weights):
        emb2char[tuple(emb)] = char
        #print(char, tuple(emb))

    for char, emb in zip(vocab, emb_weights):
        if not char.isalnum():
            continue
        neigh = nearest(emb_weights, emb, 5)
        print(repr(char), end=': ')
        print(', '.join(repr(vocab[i]) for i in neigh))


if __name__ == '__main__':
    main()
