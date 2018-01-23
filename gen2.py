#!/usr/bin/env python3

import itertools
import json
import os
import random

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

def main():
    en = []
    js = []
    vocab_en = {"EOL"}
    vocab_js = {'"add"', '"modify"', '"cancel"', '{', '}', '"command":', ',',
                '"name":', '"what":', '"delay":', '"repeat":'}

    for l in (add_words, modify_words, cancel_words, delay_words, repeat_words):
        for w in " ".join(l).split():
            vocab_en.add(w)
    for _ in range(20000):
        msg_len = int(random.gauss(10, 2))
        msg = '"'+''.join(random.choice(chars) for _ in range(msg_len))+'"'
        #vocab_en.add(msg)
        #vocab_js.add(msg)
        op = random.choice(["add", "modify", "cancel"])
        txt = random.choice({"add": add_words,
                             "modify": modify_words,
                             "cancel": cancel_words}[op])

        js_data = ['"command": "'+op+'"',
                   '"name": '+msg]

        en_data = [txt, msg]

        if op == 'modify':
            en_data.append('to')
            en_data.append(random.choice(add_words))

        if op in ("add", "modify"):
            js_data.append('"what": '+msg) 

            if random.choice((True, False)):
                delay = int(random.gauss(100, 10))
                en_data.append(random.choice(delay_words) + " {} seconds".format(delay))
                js_data.append('"delay": {}'.format(delay))
                vocab_en.add(str(delay))
                vocab_js.add(str(delay))
            else:
                delay = None

            if random.choice((True, False)) or (op == 'modify' and delay is None):
                repeat = int(random.gauss(100, 10))
                en_data.append(random.choice(repeat_words) + " {} seconds".format(repeat))
                js_data.append('"repeat": {}'.format(repeat))
                vocab_en.add(str(repeat))
                vocab_js.add(str(repeat))

        en_data.append("EOL")
        js.append(' , '.join(js_data).replace("  ", " ").strip())
        en.append(' '.join(en_data).replace("  ", " ").strip())

    this_dir = os.path.dirname(__file__)
    data_dir = os.path.join(this_dir, 'gen_data')

    ven = os.path.join(data_dir, "vocab.en")
    vjs = os.path.join(data_dir, "vocab.js")
    with open(ven, "w") as fen, open(vjs, "w") as fjs:
        for w in vocab_en:
            print(w, file=fen)
        for w in vocab_js:
            print(w, file=fjs)

    train_en, test_en, train_js, test_js = train_test_split(en, js, train_size=0.6)
    dev_en, test_en, dev_js, test_js = train_test_split(test_en, test_js, train_size=0.5)

    for prefix, data_en, data_js in (('train', train_en, train_js),
                                     ('dev', dev_en, dev_js),
                                     ('tst', test_en, test_js)):
        fen = os.path.join(data_dir, prefix+".en")
        fjs = os.path.join(data_dir, prefix+".js")
        with open(fen, "w") as fen, open(fjs, "w") as fjs:
            for p in data_en:
                print(p, file=fen)
            for p in data_js:
                print(p, file=fjs)

if __name__ == '__main__':
    main()
