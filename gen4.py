#!/usr/bin/env python3

import os
import random

from rivescript.rivescript import RiveScript
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


chars = [chr(x + ord('a')) for x in range(26)]
chars.extend([chr(x + ord('0')) for x in range(10)])

add_words = ["print", "send", "emit"]
modify_words = ["modify", "change", "alter"]
cancel_words = ["cancel", "stop"]
delay_words = ["in", "after"]
repeat_words = ["every"]

for w in add_words.copy():
    add_words.append("start {}ing".format(w))
    cancel_words.append("stop {}ing".format(w))


def make_sample(rs, cls, *args, **kwargs):
    tokens = [cls] + list(args)
    for k, v in kwargs.items():
        tokens.append(k)
        tokens.append(v)
    tr_string = ' '.join(map(str, tokens))
    result = rs.reply('', tr_string)
    cmd = ['command', cls]
    en = []
    tags = []
    for w in result.split():
        if '|' in w:
            w, t = w.split('|')
            cmd.extend([t, w])
        else:
            t = 'O'
        en.append(w)
        tags.append(t)
    return cmd, en, tags


def make_vocab(lines):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(lines)
    return list(vectorizer.vocabulary_.keys())


def create_tagger_training_data(en, tags):
    result = []
    for e, tg in zip(en, tags):
        result.extend('{} {}'.format(w, t) for w, t in zip(e, tg))
        result.append('')
    return result[:-1]

COUNT=1000


def main():
    this_dir = os.path.dirname(__file__)
    data_dir = os.path.join(this_dir, 'gen_data')
    tagger_dir = os.path.join(this_dir, 'sequence_tagging')
    tagger_data_dir = os.path.join(tagger_dir, 'data')

    en = []
    pa = []
    tags = []
    def add_sample(sample):
        p, e, t = sample
        en.append(e)
        pa.append(p)
        tags.append(t)

    rs = RiveScript(utf8=True)
    rs.load_directory(os.path.join(this_dir, 'human_train_1'))
    rs.sort_replies()

    for c in ('yes', 'no', 'ping'):
        for _ in range(COUNT):
            add_sample(make_sample(rs, c))

    for _ in range(COUNT):
        count = int(random.gauss(15, 5))
        add_sample(make_sample(rs, 'maki-uchi-log', count))

    tagger_data = create_tagger_training_data(en, tags)
    with open(os.path.join(tagger_data_dir, 'test.txt'), 'w') as tf:
        for td in tagger_data:
            print(td, file=tf)

    en_lines = [' '.join(e) for e in en]
    ven = os.path.join(data_dir, "vocab.en")
    with open(ven, "w") as fen:
        for w in make_vocab(en_lines):
            print(w, file=fen)

    pa_lines = [' '.join(p) for p in pa]
    vpa = os.path.join(data_dir, "vocab.pa")
    with open(vpa, "w") as fpa:
        for w in make_vocab(pa_lines):
            print(w, file=fpa)

    train_en, test_en, train_pa, test_pa = train_test_split(en_lines, pa_lines, train_size=0.6)
    dev_en, test_en, dev_pa, test_pa = train_test_split(test_en, test_pa, train_size=0.5)

    for prefix, data_en, data_pa in (('train', train_en, train_pa),
                                     ('dev', dev_en, dev_pa),
                                     ('tst', test_en, test_pa)):
        fen = os.path.join(data_dir, prefix+".en")
        fpa = os.path.join(data_dir, prefix+".pa")
        with open(fen, "w") as fen, open(fpa, "w") as fpa:
            for p in data_en:
                print(p, file=fen)
            for p in data_pa:
                print(p, file=fpa)

if __name__ == '__main__':
    main()
