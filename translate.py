#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
import tensorflow as tf

from nmt.nmt import attention_model, gnmt_model, model_helper
from nmt.nmt.nmt import add_arguments, create_hparams, ensure_compatible_hparams
from nmt.nmt.utils import nmt_utils
from nmt.nmt.utils.misc_utils import get_config_proto, load_hparams
from sequence_tagging.model.config import Config
from sequence_tagging.model.ner_model import NERModel


def make_result(sentence, tags, words):
    result = {}
    vals = dict((t, w) for t, w in zip(tags, words) if t != 'O')
    nwords = sentence.split()
    for k, v in zip(nwords[::2], nwords[1::2]):
        if v in vals:
            v = vals[v]
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    if v in ("True", "False"):
                        v = v == "True"
                    else:
                        v = v.strip('"\'')
        result[k] = v
    return result


class MyTranslator:

    def __init__(self, tagger):
        self._tagger = tagger

    def set_nmt_model_dir(self, nmt_model_dir):
        self._ckpt = tf.train.latest_checkpoint(nmt_model_dir)
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        flags = parser.parse_args([])
        self._hparams = load_hparams(nmt_model_dir)
        if self._hparams.attention_architecture == "standard":
            model_creator = attention_model.AttentionModel
        elif self._hparams.attention_architecture in ["gnmt", "gnmt_v2"]:
            model_creator = gnmt_model.GNMTModel
        self._infer_model = model_helper.create_infer_model(model_creator, self._hparams)

        self._sess = tf.InteractiveSession(graph=self._infer_model.graph, config=get_config_proto())
        self._sess.run(tf.tables_initializer())
        self._infer_model.model.saver.restore(self._sess, self._ckpt)

    def tag(self, words):
        return self._tagger.predict(words)

    def infer(self, sentence):
        self._sess.run(self._infer_model.iterator.initializer,
                       feed_dict={
                           self._infer_model.src_placeholder: [sentence],
                           self._infer_model.batch_size_placeholder: 1,
                       })

        nmt_outputs, _ = self._infer_model.model.decode(self._sess)

        if self._hparams.beam_width == 0:
            nmt_outputs = np.expand_dims(nmt_outputs, 0)

        translation = nmt_utils.get_translation(
            nmt_outputs[0],
            0,
            tgt_eos=self._hparams.eos,
            subword_option=self._hparams.subword_option)
        return translation.decode()

    def translate(self, sentence):
        words = sentence.strip().split()
        tags = self.tag(words)
        replaced = ' '.join(w if t == 'O' else t for w, t in zip(words, tags))
        translated = self.infer(replaced)
        return make_result(translated, tags, words)


def make_translator():
    this_dir = os.path.dirname(__file__)
    cwd = os.getcwd()
    os.chdir(os.path.join(this_dir, 'sequence_tagging'))
    tg_config = Config()
    tg_model = NERModel(tg_config)
    tg_model.build()
    tg_model.restore_session(tg_config.dir_model)
    os.chdir(cwd)
    result = MyTranslator(tg_model)
    result.set_nmt_model_dir(os.path.join(this_dir, 'sched_model'))
    return result
    

def main():
    translator = make_translator()
    sentence = ' '.join(sys.argv[1:])
    print(translator.translate(sentence))
    

if __name__ == '__main__':
    main()
