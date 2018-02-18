#!/usr/bin/env python3
import os
import pickle

from keras.models import load_model

from translator import Translator


def main():
    this_dir = os.path.dirname(__file__)
    data_dir = os.path.join(this_dir, "gen_data")

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
