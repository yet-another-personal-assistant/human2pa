#!/usr/bin/env python3

import argparse
import csv
import os
import sys

from bs4 import BeautifulSoup, NavigableString
import nltk


phrases_file = 'data/input.txt'


def profile_as_list(profile):
    result = []
    p1s = BeautifulSoup(profile, 'html.parser')
    for p in p1s.span.children:
        if isinstance(p, NavigableString):
            result.append(p)
    return result


def parse_dialogue(dialogue):
    d = BeautifulSoup(dialogue, 'html.parser')
    for c in d.find_all('span'):
        yield c['class'][0], c.text.split(':', 2)[1].strip()


def load_profiles(profiles_file, labels=None):
    filter = set()
    if labels is not None:
        for i, label in enumerate(labels):
            if label == '+':
                print("adding user", i)
                filter.add(i)
    profiles = []
    with open(profiles_file) as tsv:
        tsvin = csv.reader(tsv, delimiter='\t')
        for i, row in enumerate(tsvin):
            if labels is None or i in filter:
                profiles.append(row[:-1])
    return profiles


def load_labels(labels_file_path):
    with open(os.path.expanduser(labels_file_path)) as labels_file:
        return [l.strip() for l in labels_file.readlines()]


def profile(args):
    profiles = load_profiles(os.path.expanduser(args.profiles))
    labels = load_labels(os.path.expanduser(args.labels))
    indices = []
    if args.search is not None:
        for i, profile in enumerate(profiles):
            for line in profile:
                if args.search in line:
                    indices.append(i)
                    break
    elif args.index is not None:
        indices.append(args.index)
    for index in indices:
        print(index, labels[index])
        for line in profiles[index]:
            print("\t", line)


def label(args):
    profiles_file = os.path.expanduser(args.profiles)
    dialogues_file = os.path.expanduser(args.dialogues)
    profiles = load_profiles(profiles_file)
    profile_indices = {tuple(profile):i for i, profile in enumerate(profiles)}
    if args.labels is None:
        labels = ['?' for profile in profiles]
    else:
        labels = load_labels(os.path.expanduser(args.labels))
    print(profile_indices)
    with open(dialogues_file) as tsv:
        tsvin = csv.DictReader(tsv, delimiter='\t')
        for n, record in enumerate(tsvin):
            if n < args.skip:
                continue
            names = {}
            indices = [0, 0, 0]
            print("Dialogue", n+1)
            for i in (1, 2):
                profile = record['persona_{}_profile'.format(i)]
                as_list = profile_as_list(profile)
                idx = profile_indices[tuple(as_list)]
                print("Profile", idx)
                print("\t"+"\n\t".join(as_list))
                names['participant_'+str(i)] = '{} {}'.format(str(idx), labels[idx])
                indices[i] = idx
            for participant, phrase in parse_dialogue(record['dialogue']):
                print(names[participant], '\t:', phrase)
            for i in (1, 2):
                idx = indices[i]
                while True:
                    label = input("Label for user {} [{}]: ".format(idx, labels[idx]))
                    if not label:
                        label = labels[idx]
                    if label in '+-?':
                        break
                labels[idx] = label
            with open('labels.txt', 'w') as labels_file:
                for label in labels:
                    print(label, file=labels_file)


def extract(dialogues_file, output_file):
    stdout = sys.stdout
    if output_file is not None and output_file != '-':
        sys.stdout = open(output_file, "w")
    with open(dialogues_file) as tsv:
        tsv.readline()
        tsvin = csv.reader(tsv, delimiter='\t')
        for profile1, profile2, dialogue in tsvin:
            for participant, phrase in parse_dialogue(dialogue):
                print(phrase)
    sys.stdout = stdout


def extract_words(dialogues_file, output_file):
    stdout = sys.stdout
    if output_file is not None and output_file != '-':
        sys.stdout = open(output_file, "w")
    allwords = set()
    with open(dialogues_file) as tsv:
        tsv.readline()
        tsvin = csv.reader(tsv, delimiter='\t')
        for profile1, profile2, dialogue in tsvin:
            for participant, phrase in parse_dialogue(dialogue):
                for word in nltk.tokenize.word_tokenize(phrase):
                    allwords.add(word)
    for word in sorted(allwords):
        print(word)
    sys.stdout = stdout


def main(profiles_file, dialogues_file, args):
    if args.labels is None:
        profiles = load_profiles(profiles_file)
    else:
        labels = load_labels(args.labels)
        profiles = load_profiles(profiles_file, labels)
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
            for participant, phrase in parse_dialogue(dialogue):
                if participant in use:
                    print(phrase, file=out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--profiles')
    parser.add_argument('-d', '--dialogues', required=True)
    parser.add_argument('-l', '--labels')
    subparsers = parser.add_subparsers(dest='command')
    label_parser = subparsers.add_parser("label")
    label_parser.add_argument('--skip', type=int, default=0)
    profile_parser = subparsers.add_parser("profile")
    profile_parser.add_argument('--index', type=int)
    profile_parser.add_argument('--search')
    generate_parser = subparsers.add_parser("generate")

    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("-o", "--output")

    extract_w_parser = subparsers.add_parser("extract-words")
    extract_w_parser.add_argument("-o", "--output")

    args = parser.parse_args()
    if args.command == 'label':
        label(args)
    elif args.command == 'profile':
        profile(args)
    elif args.command == 'extract':
        extract(args.dialogues, args.output)
    elif args.command == 'extract-words':
        extract_words(args.dialogues, args.output)
    else:
        main(os.path.expanduser(args.profiles),
             os.path.expanduser(args.dialogues),
             args)
