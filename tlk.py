#!/usr/bin/env python3

import argparse
import csv
import os

from bs4 import BeautifulSoup, NavigableString


phrases_file = 'data/input.txt'


def profile_as_list(profile):
    result = []
    p1s = BeautifulSoup(profile, 'html.parser')
    for p in p1s.span.children:
        if isinstance(p, NavigableString):
            result.append(p)
    return result


def main(profiles_file, dialogues_file, filter_file):
    filter = set()
    if filter_file is not None:
        with open(filter_file) as f:
            for l in f:
                if l.startswith('======'):
                    break
                if not l.startswith('\t'):
                    filter.add(int(l.split()[0]))
    profiles = []
    with open(profiles_file) as tsv:
        tsvin = csv.reader(tsv, delimiter='\t')
        for i, row in enumerate(tsvin):
            if not filter:
                print(i, '\t', row[0])
                for r in row[1:]:
                    print('\t', r)
            elif i in filter:
                profiles.append(row[:-1])
    with open(dialogues_file) as tsv, open(phrases_file, 'w') as out:
        tsv.readline()
        tsvin = csv.reader(tsv, delimiter='\t')
        for profile1, profile2, dialogue in tsvin:
            use = set()
            if profile_as_list(profile1) in profiles:
                use.add('participant_1')
            if profile_as_list(profile2) in profiles:
                use.add('participant_2')
            if not use:
                continue
            d = BeautifulSoup(dialogue, 'html.parser')
            for c in d.find_all('span'):
                if c['class'][0] in use:
                    print(c.text.split(':', 2)[1].strip(), file=out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--profiles', required=True)
    parser.add_argument('-d', '--dialogues', required=True)
    parser.add_argument('-f', '--filter')
    args = parser.parse_args()
    main(os.path.expanduser(args.profiles),
         os.path.expanduser(args.dialogues),
         os.path.expanduser(args.filter) if args.filter else None)
