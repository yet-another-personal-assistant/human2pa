#!/usr/bin/env python3
import random
import re

from pathlib import Path

from rivescript.rivescript import RiveScript


tag_var_re = re.compile(r'data-([a-z-]+)\((.*?)\)|(\S+)')


def make_sample(rs, cls, *args, **kwargs):
    tokens = [cls] + list(args)
    for k, v in kwargs.items():
        tokens.append(k)
        tokens.append(v)
    result = rs.reply('', ' '.join(map(str, tokens))).strip()
    if result == '[ERR: No Reply Matched]':
        raise Exception("failed to generate string for {}".format(tokens))
    cmd, en, tags = cls, [], []
    for tag, value, just_word in tag_var_re.findall(result):
        if just_word:
            en.append(just_word)
            tags.append("O")
        else:
            _, tag = tag.split('-', maxsplit=1)
            words = value.split()
            ntags = [f"I-{tag}" for _ in words]
            if len(words) > 1:
                ntags[0][0] = "B"
            en.extend(words)
            tags.extend(ntags)
    return cmd, en, tags


def generate(rive, data_dir=Path('.'), count=1000, counts=None):
    if counts is None:
        counts = {
            'yes':  count // 4,
            'no': count // 4,
            'ping': count // 2,
            'acknowledge': count // 2,
            'makiuchi': count,
        }

    human, pa, ner = [], [], []
    def add_sample(sample):
        p, h, r = sample
        human.append(h)
        pa.append(p)
        ner.append(r)

    rs = RiveScript(utf8=True)
    rs.load_directory(str(Path('.') / rive))
    rs.sort_replies()

    for intent, count in counts.items():
        for _ in range(count):
            args = ()
            if intent == 'makiuchi':
                args = (max(int(random.gauss(15, 5)), 1),)
            add_sample(make_sample(rs, intent, *args))

    with open(data_dir / 'labels.txt', "w") as labels:
        for label, text in zip(pa, human):
            print(f"__label__{label}", " ".join(text), file=labels)

    with open(data_dir / 'tags.txt', "w") as tags:
        for text, ner_tags in zip(human, ner):
            for word, tag in zip(text, ner_tags):
                print(word, tag, file=tags)
            print(file=tags)
