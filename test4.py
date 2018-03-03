#!/usr/bin/env python3
import glob
import os
import pickle
import pprint

import numpy as np

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from rasa_nlu import registry
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Metadata, Interpreter

from translator import Translator


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

    config = RasaNLUConfig()
    config.pipeline = registry.registered_pipeline_templates["spacy_sklearn"]
    config.max_training_processes = 4
    model_dir = glob.glob(data_dir+"/rasa/default/model_*")[0]

    interpreter = Interpreter.load(model_dir, config)

    tst_en = os.path.join(data_dir, "train.en")
    tst_tg = os.path.join(data_dir, "train.tg")
    with open(tst_en) as fen, open(tst_tg) as ftg:
        for en, tg in zip(fen.readlines(), ftg.readlines()):
            en = en.strip()
            print(en)
            parsed = interpreter.parse(en)
            result = [parsed['intent_ranking'][0]['name']]
            for entity in parsed['entities']:
                result.append(entity['entity']+':')
                result.append('"'+entity['value']+'"')
            print(' '.join(result))
            print("expected", tg.split())
            print()

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
