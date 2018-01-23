#!/usr/bin/env python3

import itertools
import json
import os
import random
import re

from sklearn.model_selection import train_test_split

words=set('''hello world test post rest split word words best least
last mirror bird dog cat gold science python teach go play computer task
meat food stack light fly bag people human bot nice bad linear white
black green gold'''.split())

mre = re.compile(r'([{}:])')

def ph2js(p):
    result = json.dumps({"unknown": " {} ".format(p)})
    return re.sub(mre, r' \1 ', result).replace("  ", " ").strip()

this_dir = os.path.dirname(__file__)
data_dir = os.path.join(this_dir, 'gen_data')

ven = os.path.join(data_dir, "vocab.en")
vjs = os.path.join(data_dir, "vocab.js")
with open(ven, "w") as en, open(vjs, "w") as js:
    for add in ['{', '}', '"', ':', '"unknown"', 'END']:
        if not add in words:
            print(add, file=en)
            print(add, file=js)
    for w in words:
        print(w, file=en)
        print(w, file=js)

phrases = []
for i in range(1, 5):
    phrases.extend(" ".join(p) for p in itertools.permutations(words, i))

phrases = list(p for p in random.sample(phrases, 200000))
train, test = train_test_split(phrases, train_size=0.6)
dev, test = train_test_split(test, train_size=0.5)

for prefix, data in (('train', train), ('dev', dev), ('tst', test)):
    fen = os.path.join(data_dir, prefix+".en")
    fjs = os.path.join(data_dir, prefix+".js")
    with open(fen, "w") as en, open(fjs, "w") as js:
        for p in data:
            print(p, 'END', file=en)
            print(ph2js(p), file=js)
