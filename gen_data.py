#!/usr/bin/env python3
import json
import os
import random
import re

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
    cmd, en, rasa_entities = cls, [], []
    for tag, value, just_word in tag_var_re.findall(result):
        if just_word:
            en.append(just_word)
        else:
            _, tag = tag.split('-', maxsplit=1)
            words = value.split()
            start = len(' '.join(en))
            if en:
                start += 1
                en.extend(words)
                end = len(' '.join(en))
                rasa_entities.append({"start": start, "end": end,
                                      "value": value, "entity": tag})
                assert ' '.join(en)[start:end] == value
    return cmd, en, rasa_entities

COUNT=1000

def main():
    this_dir = os.path.dirname(__file__)
    data_dir = os.path.join(this_dir, 'gen_data')
    tagger_dir = os.path.join(this_dir, 'sequence_tagging')
    tagger_data_dir = os.path.join(tagger_dir, 'data')

    en, pa, rasa = [], [], []
    def add_sample(sample):
        p, e, r = sample
        en.append(e)
        pa.append(p)
        rasa.append(r)

    rs = RiveScript(utf8=True)
    rs.load_directory(os.path.join(this_dir, 'en_training'))
    rs.sort_replies()

    for c in ('yes', 'no'):
        for _ in range(COUNT//4):
            add_sample(make_sample(rs, c))
    for c in ('ping', 'acknowledge'):
        for _ in range(COUNT//2):
            add_sample(make_sample(rs, c))

    for _ in range(COUNT):
        count = max(int(random.gauss(15, 5)), 1)
        add_sample(make_sample(rs, 'makiuchi', count))

    rasa_examples = []
    for e, p, r in zip(en, pa, rasa):
        sample = {"text": ' '.join(e), "intent": p}
        if r:
            sample["entities"] = r
            rasa_examples.append(sample)

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    with open(os.path.join(data_dir, "rasa_train.js"), "w") as rf:
        json.dump({"rasa_nlu_data": {"common_examples": rasa_examples,
                                     "regex_features": [],
                                     "entity_synonims": []}},
                  rf)

if __name__ == '__main__':
    main()
