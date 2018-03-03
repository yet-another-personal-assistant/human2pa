#!/usr/bin/env python3
import glob
import os
import pickle
import pprint
import sys

from keras.models import load_model
from rasa_nlu import registry
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Metadata, Interpreter

from translator import Translator


def main():
    this_dir = os.path.dirname(__file__)
    data_dir = os.path.join(this_dir, "gen_data")

    #classifier = load_model(os.path.join(data_dir, "trained.cls"))
    #tagger = load_model(os.path.join(data_dir, "trained.tagger"))
    #with open(os.path.join(data_dir, "trained.lb"), 'rb') as labels_file:
    #    lb = pickle.load(labels_file)

    #translator = Translator(lb=lb, cls=classifier)

    line = ' '.join(sys.argv[1:])
    print(line)
    #print(translator.classify2(line))
    config = RasaNLUConfig()
    config.pipeline = registry.registered_pipeline_templates["spacy_sklearn"]
    config.max_training_processes = 4
    model_dir = glob.glob(data_dir+"/rasa/default/model_*")[0]

    interpreter = Interpreter.load(model_dir, config)
    parsed = interpreter.parse(line)
    intent = parsed['intent_ranking'][0]['name']
    result = [intent]
    for entity in parsed['entities']:
    #    if entity['intent'] == intent:
        result.append(entity['entity']+':')
    #    else:
    #        result.append(entity['intent']+'-'+entity['entity']+':')
        result.append('"'+entity['value']+'"')
    print(' '.join(result))
    pprint.pprint(parsed)


if __name__ == '__main__':
    main()
