import numpy as np


class CharEmbed:

    def __init__(self, vocab=None, data=None):
        if vocab is None:
            vocab = set(data)
        self.idx2char = ['']+list(vocab)
        self.vocab_size = len(self.idx2char)
        self.char2idx = {c: i for i, c in enumerate(self.idx2char)}

    def onehot(self, char):
        result = np.zeros(self.vocab_size)
        result[self.char2idx[char]] = 1
        return result
