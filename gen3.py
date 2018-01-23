#!/usr/bin/env python3

import os
import random

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


def make_sched_sample():
    msg_len = int(random.gauss(10, 2))
    msg = ''.join(random.choice(chars) for _ in range(msg_len))
    op = random.choice(["add", "modify", "cancel"])
    txt = random.choice({"add": add_words,
                         "modify": modify_words,
                         "cancel": cancel_words}[op])

    js_data = {"command": op, "name": msg, "what": msg}
    cmd_data = txt.split()
    cmd_data.append('"'+msg+'"')
    cmd_tags = ['O'] * len(cmd_data)
    cmd_tags[-1] = 'data_sched_name'

    if op == 'modify':
        cmd_data.append('to')
        cmd_data.extend(random.choice(add_words).split())
        cmd_tags.extend(['O'] * (len(cmd_data) - len(cmd_tags)))

    if op in ("add", "modify"):
        js_data["what"] = msg

        if random.choice((True, False)):
            delay = int(random.gauss(100, 10))
            delay_data = []
            delay_data.extend(random.choice(delay_words).split()),
            delay_data.append(str(delay))
            delay_data.append('seconds')
            delay_tags = ['O'] * len(delay_data)
            delay_tags[-2] = 'data_sched_delay'
            js_data["delay"] = delay
        else:
            delay_data = delay_tags = []

        if random.choice((True, False)) or (op == 'modify' and not delay_data):
            repeat = int(random.gauss(100, 10))
            repeat_data = []
            repeat_data.extend(random.choice(repeat_words).split()),
            repeat_data.append(str(repeat))
            repeat_data.append('seconds')
            repeat_tags = ['O'] * len(repeat_data)
            repeat_tags[-2] = 'data_sched_repeat'
            js_data["repeat"] = repeat
        else:
            repeat_data = repeat_tags = []

        if op == 'modify' or cmd_data[0] == 'start':
            en_data = cmd_data
            tags = cmd_tags
            l = list(zip((delay_data, repeat_data),
                         (delay_tags, repeat_tags)))
        else:
            en_data = []
            tags = []
            l = list(zip((cmd_data, delay_data, repeat_data),
                         (cmd_tags, delay_tags, repeat_tags)))

        random.shuffle(l)
        for d, t in l:
            en_data.extend(d)
            tags.extend(t)
    else:
        en_data, tags = cmd_data, cmd_tags

    if random.choice((True, False)):
        if random.choice((True, False)):
            en_data.extend([',', 'please'])
            tags.extend(['O', 'data_tone'])
            js_data['tone'] = "please"
        else:
            en_data.insert(0, ',')
            en_data.insert(0, 'please')
            tags.insert(0, 'O')
            tags.insert(0, 'data_tone')
            js_data['tone'] = "please"

    return js_data, en_data, tags


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


def main():
    this_dir = os.path.dirname(__file__)
    data_dir = os.path.join(this_dir, 'gen_data')
    tagger_dir = os.path.join(this_dir, 'sequence_tagging')
    tagger_data_dir = os.path.join(tagger_dir, 'data')

    en = []
    js = []
    tags = []

    for _ in range(100):
        j, e, t = make_sched_sample()
        en.append(e)
        js.append(j)
        tags.append(t)

    tagger_data = create_tagger_training_data(en, tags)
    with open(os.path.join(tagger_data_dir, 'test.txt'), 'w') as tf:
        for td in tagger_data:
            print(td, file=tf)

    en_lines = []
    for e, tg in zip(en, tags):
        line = [w if t == 'O' else t for w, t in zip(e, tg)]
        en_lines.append(' '.join(line))
    ven = os.path.join(data_dir, "vocab.en")
    with open(ven, "w") as fen:
        for w in make_vocab(en_lines):
            print(w, file=fen)

    js_lines = []
    for j in js:
        line = ['command', j['command'],
                'name', 'data_sched_name',
                'what', 'data_sched_name']
        for f in ('repeat', 'delay'):
            if f in j:
                line.append(f)
                line.append('data_sched_' + f)
        if 'tone' in j:
            line.extend(['tone', 'data_tone'])
        js_lines.append(' '.join(line))
    vjs = os.path.join(data_dir, "vocab.js")
    with open(vjs, "w") as fjs:
        for w in make_vocab(js_lines):
            print(w, file=fjs)

    train_en, test_en, train_js, test_js = train_test_split(en_lines, js_lines, train_size=0.6)
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
