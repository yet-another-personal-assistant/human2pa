import os
import pickle

import numpy as np

from keras.layers import Input
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences


class Seq2Seq:

    def __init__(self, data_dir):
        self.model = load_model(os.path.join(data_dir, "s2s.h5"))
        self.encoder = encoder_from_s2s(self.model)
        self.decoder = decoder_from_s2s(self.model)
        with open(os.path.join(data_dir, "tag.lb"), 'rb') as tag_binarizer:
            self.tb = pickle.load(tag_binarizer)

        with open(os.path.join(data_dir, "chars")) as chars:
            encoder_chars = list(chars.readline())

        self.enc_idx = dict((v, i) for i, v in enumerate(encoder_chars))

    def encode_sequence(self, input_seq):
        encoder_input_data = np.zeros((1, 100, 100), dtype='float32')
        for t, char in enumerate(input_seq.strip()):
            encoder_input_data[0, t, self.enc_idx[char]] = 1.
        return encoder_input_data

    def decode_sequence(self, input_seq):
        # Encode the input as state vectors.
        states_value = self.encoder.predict(input_seq)

        # Populate the first character of target sequence with the start character.
        target_seq = pad_sequences([self.tb.transform(['START'])], padding='post', maxlen=30)

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = []
        while not stop_condition:
            output_tokens, h, c = self.decoder.predict([target_seq] + states_value)

            # Sample a token
            result = self.tb.inverse_transform(output_tokens[0])[0]

            # Exit condition: either hit max length
            # or find stop character.
            if (result == 'END' or len(decoded_sentence) > 30):
                stop_condition = True
            else:
                decoded_sentence.append(result)

            # Update the target sequence (of length 1).
            target_seq = pad_sequences([self.tb.transform([result])], padding='post', maxlen=30)

            # Update states
            states_value = [h, c]

        return decoded_sentence

    def translate(self, line):
        return self.decode_sequence(self.encode_sequence(line))


def encoder_from_s2s(model):
    latent_dim = model.layers[2].output_shape[0][1]

    encoder_inputs = model.input[0]   # input_1
    encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output   # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    return Model(encoder_inputs, encoder_states)


def decoder_from_s2s(model):
    latent_dim = model.layers[2].output_shape[0][1]

    decoder_inputs = model.input[1]   # input_2
    decoder_state_input_h = Input(shape=(latent_dim,), name='input_3')
    decoder_state_input_c = Input(shape=(latent_dim,), name='input_4')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.layers[3]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_inputs,
                                                             initial_state=decoder_states_inputs)
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)
    return Model([decoder_inputs] + decoder_states_inputs,
                 [decoder_outputs] + decoder_states)


def load_sentences(filename):
    with open(filename) as f:
        return [l.strip() for l in f.readlines()]

