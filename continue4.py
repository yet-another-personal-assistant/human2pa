#!/usr/bin/env python3
import os
import pickle

import numpy as np

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from utils import load_sentences


def train_tagger(model, tb, sentences, tags, enc_idx):
    num_samples = len(sentences)  # Number of samples to train on.
    num_encoder_tokens = 100
    num_decoder_tokens = len(tb.classes_)
    max_encoder_seq_length = 100
    max_decoder_seq_length = 30

    input_texts = sentences
    target_texts = tags
    target_tokens = tb.classes_

    input_token_index = enc_idx

    encoder_input_data = np.zeros(
        (num_samples, max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')

    for i, input_text in enumerate(input_texts):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.

    # add one extra element at the end
    decoder_data = pad_sequences([encode_tag(tb, target) for target in target_texts],
                                 padding='post', maxlen=max_decoder_seq_length+1)
    decoder_target_data = decoder_data[:,1:,:]
    decoder_input_data = decoder_data[:,:-1,:]

    for i in range(10):
        model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                  batch_size=128,
                  epochs=10,
                  verbose=1,
                  validation_split=0.2)
        print("saving after step", i+1)
        model.save("gen_data/s2s.h5")


def encode_tag(tb, line):
    return tb.transform(['START'] + line.split() + ['END'])


def main():
    this_dir = os.path.dirname(__file__)
    data_dir = os.path.join(this_dir, "gen_data")

    sentences = load_sentences(os.path.join(data_dir, "train.en"))
    tags = load_sentences(os.path.join(data_dir, "train.tg"))

    test_sentences = load_sentences(os.path.join(data_dir, "dev.en"))
    test_tags = load_sentences(os.path.join(data_dir, "dev.tg"))

    s2s_name = os.path.join(data_dir, "s2s.h5")
    tagger = load_model(s2s_name)
    
    with open(os.path.join(data_dir, "tag.lb"), 'rb') as tag_binarizer:
        tb = pickle.load(tag_binarizer)

    with open(os.path.join(data_dir, "chars")) as chars:
        encoder_chars = list(chars.readline())

    enc_idx = dict((v, i) for i, v in enumerate(encoder_chars))

    train_tagger(tagger, tb, sentences+test_sentences, tags+test_tags, enc_idx)

    tagger.save(os.path.join(data_dir, 's2s.h5'))


if __name__ == '__main__':
    main()
