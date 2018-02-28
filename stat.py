#!/usr/bin/env python3
import os

from collections import Counter

def main():
    this_dir = os.path.dirname(__file__)
    data_dir = os.path.join(this_dir, 'gen_data')

    ftg = os.path.join(data_dir, "train.tg")
    fpa = os.path.join(data_dir, "train.pa")
    stat = Counter()
    lens = Counter()
    total = 0
    total1 = 0
    with open(ftg) as ftg, open(fpa) as fpa:
        for t, pa in zip(ftg, fpa):
            total += 1
            lens[len(t.strip().split())] += 1
            if t.strip() == 'O':
                total1 += 1
                a = pa.split()[0]
                stat[a] += 1
    print(total1, total)
    print(stat)
    print(lens)

if __name__ == '__main__':
    main()
