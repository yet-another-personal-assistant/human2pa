#!/usr/bin/env python3
import os
import pickle

import numpy as np

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from translator import Translator
from translator.seq2seq import Seq2Seq
from utils import encoder_from_s2s, decoder_from_s2s


def decode_sequence(input_seq, encoder_model, decoder_model, tb):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Populate the first character of target sequence with the start character.
    target_seq = pad_sequences([tb.transform(['START'])], padding='post', maxlen=30)

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        result = tb.inverse_transform(output_tokens[0])[0]

        # Exit condition: either hit max length
        # or find stop character.
        if (result == 'END' or len(decoded_sentence) > 30):
            stop_condition = True
        else:
            decoded_sentence.append(result)

        # Update the target sequence (of length 1).
        target_seq = pad_sequences([tb.transform([result])], padding='post', maxlen=30)

        # Update states
        states_value = [h, c]

    return decoded_sentence


def main():
    this_dir = os.path.dirname(__file__)
    data_dir = os.path.join(this_dir, "gen_data")

    s2s = Seq2Seq(data_dir)

    tst_en = os.path.join(data_dir, "train.en")
    tst_tg = os.path.join(data_dir, "train.tg")
    with open(tst_en) as fen, open(tst_tg) as ftg:
        count = 0
        for en, tg in zip(fen.readlines(), ftg.readlines()):
            print(en.strip())
            print("expected", tg.split())
            print("actual", s2s.translate(en))
            print()
            if count > 100:
                break
            count += 1

    return

    classifier = load_model(os.path.join(data_dir, "trained.cls"))
    with open(os.path.join(data_dir, "trained.lb"), 'rb') as labels_file:
        lb = pickle.load(labels_file)

    translator = Translator(lb=lb, cls=classifier)

    tst_en = os.path.join(data_dir, "tst.en")
    tst_pa = os.path.join(data_dir, "tst.pa")
    cnt = 0
    wrong = 0
    with open(tst_en) as fen, open(tst_pa) as fpa:
        for en, pa in zip(fen.readlines(), fpa.readlines()):
            cls = translator.classify(en)
            act = pa.split(maxsplit=1)[0]
            cnt += 1
            if cls != act:
                wrong += 1
                print(act, cls, en.strip())
    print(cnt, wrong, (cnt-wrong)/cnt*100)


if __name__ == '__main__':
    main()
