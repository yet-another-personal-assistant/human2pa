#!/usr/bin/env python3
import logging
import os
import time

from gensim.models import word2vec
from sequence_tagging.model.ner_model import NERModel
from sequence_tagging.model.data_utils import CoNLLDataset, get_processing_word
from sklearn.feature_extraction.text import TfidfVectorizer

from translator import Translator


def load_sentences(file_name):
    with open(file_name) as fen:
        return fen.readlines()


def load_labels(file_name):
    with open(file_name) as fpa:
        return [line.strip().split(maxsplit=1)[0] for line in fpa]


def test_tagger_train(sentences, labels, tags):
    count = int(len(sentences)/10)
    label_count = len(set(labels))
    translator2 = Translator(label_count)
    translator2.lb.fit(labels)
    translator2.train_tagger(sentences[:count], tags[:count])


def load_vocab(lines):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(lines)
    return vectorizer.vocabulary_


def load_ner_vocab(lines):
    vocab = set()
    for line in lines:
        vocab.update(line.split())
    return dict((w, i) for i, w in enumerate(vocab))


class MyConfig:
    dim_word = 300
    dim_char = 100
    hidden_size_char = 100 # lstm on chars
    hidden_size_lstm = 300 # lstm on word embeddings
    train_embeddings = False
    use_crf = True
    use_chars = True
    nepochs          = 15
    dropout          = 0.5
    batch_size       = 20
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.9
    clip             = -1 # if negative, no clipping
    nepoch_no_imprv  = 3

    dir_output = "results/test/"
    dir_model  = dir_output + "model.weights/"
    path_log   = dir_output + "log.txt"


    def __init__(self, sentences, tags):
        self.logger = logging.getLogger(__name__)
        new_sentences = [s.split() for s in sentences]
        new_sentences.append(['$UNK$'])
        wv_model = word2vec.Word2Vec(new_sentences, min_count=1)
        w2v = wv_model.wv
        self.embeddings = w2v.vectors
        self.vocab_words = w2v
        self.vocab_tags = load_ner_vocab(tags)
        self.ntags = len(self.vocab_tags)

        chars = set()
        for word in load_ner_vocab(sentences):
            [chars.add(c) for c in word]
            cd = {}
        for i, c in enumerate(chars):
            cd[c] = i

        self.vocab_chars = cd
        self.nchars = len(cd)
        self.processing_word = get_processing_word(self.vocab_words,
                                                   self.vocab_chars,
                                                   lowercase=True,
                                                   chars=self.use_chars)
        self.processing_tag  = get_processing_word(self.vocab_tags,
                                                   lowercase=False,
                                                   allow_unk=False)


def make_ner_model(config):
    ner_model = NERModel(config)
    ner_model.build()
    return ner_model


def make_ner_dataset(filename, sentences, tags, config):
    with open(filename, "w") as datafile:
        for s, ts in zip(sentences, tags):
            for w, t in zip(s.split(), ts.split()):
                if t not in config.vocab_tags:
                    raise Exception(t, "not in vocab tags!")
                print(w, t, file=datafile)
            print(file=datafile)

    dataset = CoNLLDataset(filename, config.processing_word,
                           config.processing_tag, None)
    return dataset


def main():
    this_dir = os.path.dirname(__file__)
    data_dir = os.path.join(this_dir, "gen_data")
    sentences = load_sentences(os.path.join(data_dir, "train.en"))
    labels = load_labels(os.path.join(data_dir, "train.pa"))
    tags = load_sentences(os.path.join(data_dir, "train.tg"))
    label_count = len(set(labels))

    translator = Translator(label_count)
    translator.lb.fit(labels)

    ner_config = MyConfig(sentences, tags)
    print(ner_config.vocab_tags)
    print(ner_config.embeddings.shape)
    ner_model = make_ner_model(ner_config)
    ner_train = make_ner_dataset(os.path.join(data_dir, "ner_train"),
                                 sentences, tags, ner_config)

    v_sentences = load_sentences(os.path.join(data_dir, "dev.en"))
    v_labels = load_labels(os.path.join(data_dir, "dev.pa"))
    v_tags = load_sentences(os.path.join(data_dir, "dev.tg"))
    ner_dev = make_ner_dataset(os.path.join(data_dir, "ner_dev"),
                               v_sentences, v_tags, ner_config)

    ner_model.train(ner_train, ner_dev)

    return

    test_tagger_train(sentences, labels, tags)


    translator.train_tagger(sentences, tags,
                            validation=(v_sentences, v_tags))

    translator.train_classifier(sentences, labels,
                                validation=(v_sentences, v_labels))

    test_sentences = load_sentences(os.path.join(data_dir, "tst.en"))
    test_labels = load_labels(os.path.join(data_dir, "tst.pa"))
    translator.classifier_eval(test_sentences, test_labels)

    test_sentences = load_sentences(os.path.join(data_dir, "dev.en"))
    test_labels = load_labels(os.path.join(data_dir, "dev.pa"))
    translator.classifier_eval(test_sentences, test_labels)

    translator.save(os.path.join(data_dir, "trained"))


if __name__ == '__main__':
    main()
