#!/usr/bin/env python3

import os
import random
import re

from rivescript.rivescript import RiveScript
from sklearn.model_selection import train_test_split


tag_var_re = re.compile(r'data-([a-z-]+)\((.*?)\)|(\S+)')

def make_sample(rs, cls, *args, **kwargs):
    tokens = [cls] + list(args)
    for k, v in kwargs.items():
        tokens.append(k)
        tokens.append(v)
    result = rs.reply('', ' '.join(map(str, tokens))).strip()
    if result == '[ERR: No Reply Matched]':
        raise Exception("failed to generate string for {}".format(tokens))
    cmd, en, tags = [cls], [], []
    for tag, value, just_word in tag_var_re.findall(result):
        if just_word:
            en.append(just_word)
            tags.append('O')
        else:
            words = value.split()
            en.append(words.pop(0))
            tags.append('B-'+tag)
            for word in words:
                en.append(word)
                tags.append('I-'+tag)
            cmd.append(tag+':')
            cmd.append('"'+value+'"')
    return cmd, en, tags


COUNT=1000

def main():
    this_dir = os.path.dirname(__file__)
    data_dir = os.path.join(this_dir, 'gen_data')
    tagger_dir = os.path.join(this_dir, 'sequence_tagging')
    tagger_data_dir = os.path.join(tagger_dir, 'data')

    en, pa, tags = [], [], []
    def add_sample(sample):
        p, e, t = sample
        en.append(e)
        pa.append(p)
        tags.append(t)

    rs = RiveScript(utf8=True)
    rs.load_directory(os.path.join(this_dir, 'human_train_1'))
    rs.sort_replies()

    for c in ('yes', 'no', 'ping', 'acknowledge', 'weather', 'status'):
        for _ in range(COUNT):
            add_sample(make_sample(rs, c))

    for _ in range(COUNT):
        count = max(int(random.gauss(15, 5)), 1)
        add_sample(make_sample(rs, 'makiuchi', count))

    to_remind = ['wash hands', 'read books', 'make tea', 'pay bills',
                 'eat food', 'buy stuff', 'take a walk', 'do maki-uchi',
                 'say hello', 'say yes', 'say no', 'play games',
                 'drink more water', 'drink some tea']

    for _ in range(COUNT):
        r = random.choice(to_remind)
        add_sample(make_sample(rs, 'remind', r))

    for _ in range(COUNT):
        r1, r2 = random.sample(to_remind, 2)
        add_sample(make_sample(rs, 'remind', r1, 'and', r2))

    things = ['puppies', 'kittens', 'food', 'mountains', 'sea',
              'wild animals', 'nature', 'abstract art', 'sky', 'flowers',
              'forest', 'wildlife', 'dragons', 'armors', 'castles',
              'tea', 'water', 'milk']
    for _ in range(COUNT):
        t = random.choice(things)
        add_sample(make_sample(rs, 'find', t))

    for _ in range(COUNT):
        t1, t2 = random.sample(things, 2)
        add_sample(make_sample(rs, 'find', t1, 'or', t2))

    for _ in range(COUNT):
        t1, t2 = random.sample(things, 2)
        add_sample(make_sample(rs, 'find', t1, 'and', t2))

    for _ in range(COUNT):
        h = int(random.gauss(8, 2))
        m = 5 * random.randrange(12)
        add_sample(make_sample(rs, 'wakeup', h, m))

    en_lines = [' '.join(e) for e in en]
    tag_lines = [' '.join(t) for t in tags]
    pa_lines = [' '.join(p) for p in pa]

    tr_e, tst_e, tr_pa, tst_pa, tr_t, tst_t = train_test_split(en_lines, pa_lines, tag_lines, train_size=0.6)
    d_e, tst_e, d_pa, tst_pa, d_t, tst_t = train_test_split(tst_e, tst_pa, tst_t, train_size=0.5)

    for prefix, data_en, data_pa, data_tg in (('train', tr_e, tr_pa, tr_t),
                                              ('dev', d_e, d_pa, d_t),
                                              ('tst', tst_e, tst_pa, tst_t)):
        fen = os.path.join(data_dir, prefix+".en")
        fpa = os.path.join(data_dir, prefix+".pa")
        ftg = os.path.join(data_dir, prefix+".tg")
        with open(fen, "w") as fen, open(fpa, "w") as fpa, open(ftg, "w") as ftg:
            for p in data_en:
                print(p, file=fen)
            for p in data_pa:
                print(p, file=fpa)
            for p in data_tg:
                print(p, file=ftg)

if __name__ == '__main__':
    main()
