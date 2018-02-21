#!/usr/bin/env python3
import itertools
import logging
import os
import pickle
import time

import numpy as np

from keras.layers import Input, LSTM, Dense
from keras.models import Model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.sequence import pad_sequences

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


def make_tagger(tags):
    latent_dim = 256
    num_encoder_tokens = 100
    max_encoder_seq_length = 100
    max_decoder_seq_length = 30

    tag_vocab = set(itertools.chain.from_iterable(t.split() for t in tags))
    tag_vocab.add('START')
    tag_vocab.add('END')
    tb = LabelBinarizer()
    tb.fit(list(tag_vocab))
    num_decoder_tokens = len(tb.classes_)

    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # Run training
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model, tb


def make_tagger_chars(sentences):
    input_characters = set(itertools.chain(*sentences))
    input_characters = sorted(list(input_characters))
    return dict((char, i) for i, char in enumerate(input_characters)), input_characters


def train_tagger(model, tb, sentences, tags, enc_idx):
    num_samples = len(sentences)  # Number of samples to train on.
    num_encoder_tokens = 100
    num_decoder_tokens = len(tb.classes_)
    max_encoder_seq_length = 100
    max_decoder_seq_length = 30

    input_texts = sentences
    target_texts = tags
    target_tokens = tb.classes_

    encoder_input_data = np.zeros(
        (num_samples, max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')

    for i, input_text in enumerate(input_texts):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, enc_idx[char]] = 1.

    # add one extra element at the end
    decoder_data = pad_sequences([encode_tag(tb, target) for target in target_texts],
                                 padding='post', maxlen=max_decoder_seq_length+1)
    decoder_target_data = decoder_data[:,1:,:]
    decoder_input_data = decoder_data[:,:-1,:]

    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=64,
              epochs=100,
              verbose=1,
              validation_split=0.2)

def encode_tag(tb, line):
    return tb.transform(['START'] + line.split() + ['END'])


def main():
    this_dir = os.path.dirname(__file__)
    data_dir = os.path.join(this_dir, "gen_data")
    sentences = load_sentences(os.path.join(data_dir, "train.en"))
    labels = load_labels(os.path.join(data_dir, "train.pa"))
    tags = load_sentences(os.path.join(data_dir, "train.tg"))
    label_count = len(set(labels))

    translator = Translator(label_count)
    translator.lb.fit(labels)

    test_sentences = load_sentences(os.path.join(data_dir, "dev.en"))
    test_tags = load_sentences(os.path.join(data_dir, "dev.tg"))

    enc_idx, enc_chars = make_tagger_chars(sentences+test_sentences)
    tagger, encoder, decoder, tb = make_tagger(tags+test_tags)
    train_tagger(tagger, tb, sentences+test_sentences, tags+test_tags, enc_idx)

    tagger.save(os.path.join(data_dir, 's2s.h5'))
    encoder.save(os.path.join(data_dir, 'encoder.h5'))
    decoder.save(os.path.join(data_dir, 'decoder.h5'))
    with open(os.path.join(data_dir, "tag.lb"), "wb") as out:
        pickle.dump(tb, out)

    with open(os.path.join(data_dir, "chars"), "w") as chars:
        print(''.join(enc_chars), file=chars)

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
