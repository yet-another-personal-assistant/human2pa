import numpy as np
import pickle

from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, LSTM, Input
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import hashing_trick
from sklearn.preprocessing import LabelBinarizer


HASH_SIZE=256


def _make_classifier(input_length, vocab_size, class_count):
    result = Sequential()
    result.add(Embedding(vocab_size, 8, input_length=input_length))
    result.add(Flatten())
    result.add(Dense(class_count, activation='sigmoid'))
    result.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    return result


def _make_tagger(vocab_size, tag_vocab_size, input_length):
    return None, None, None


def _embed(sentence):
        return hashing_trick(sentence, HASH_SIZE, 'md5')


def _train(model, prep_func, train, validation=None, epochs=10, verbose=2):
    X, y = prep_func(*train)
    validation_data = None if validation is None else prep_func(*validation)
    model.fit(X, y, epochs=epochs, verbose=verbose, shuffle=False,
              validation_data=validation_data)


class Translator:

    def __init__(self, class_count=None, cls=None, lb=None, tagger=None):
        if class_count is None and lb is None and cls is None:
            raise Exception("Class count is not known")
        self.max_length = 32
        self.lb = lb or LabelBinarizer()
        if class_count is None and lb is not None:
            class_count = len(lb.classes_)
        self.classifier = cls or _make_classifier(self.max_length, HASH_SIZE, class_count)
        if tagger is None:
            self.tagger, self.encoder, self.decoder = _make_tagger(HASH_SIZE, HASH_SIZE, self.max_length)
        else:
            self.tagger, self.encoder, self.decoder = tagger

    def _prepare_classifier_data(self, lines, labels):
        X = pad_sequences([_embed(line) for line in lines],
                            padding='post', maxlen=self.max_length)
        y = self.lb.transform(labels)
        return X, y

    def _prepare_tagger_data(self, lines, tags):
        ei = pad_sequences([_embed(line) for line in lines],
                           maxlen=self.max_length)
        di = pad_sequences([_embed(line) for line in tags],
                           maxlen=self.max_length)
        do = pad_sequences([_embed(line)[1:] for line in tags],
                           maxlen=self.max_length-1)
        return [ei, di], do

    def train_classifier(self, lines, labels, validation=None):
        _train(self.classifier, self._prepare_classifier_data,
               (lines, labels), validation)

    def train_tagger(self, lines, tags, validation=None):
        _train(self.tagger, self._prepare_tagger_data,
               (lines, tags), validation, epochs=10, verbose=1)

    def classifier_eval(self, lines, labels):
        X = pad_sequences([_embed(line) for line in lines],
                            padding='post', maxlen=self.max_length)
        y = self.lb.transform(labels)
        loss, accuracy = self.classifier.evaluate(X, y)
        print(loss, accuracy*100)

    def save(self, file_name):
        self.classifier.save(file_name+".cls")
        self.tagger.save(file_name+".tagger")
        self.encoder.save(file_name+".enc")
        self.decoder.save(file_name+".dec")
        with open(file_name+".lb", "wb") as out:
            pickle.dump(self.lb, out)

    def _classifier_predict(self, line):
        X = pad_sequences([_embed(line)],
                          padding='post', maxlen=self.max_length)
        return self.classifier.predict(X)

    def classify(self, line):
        res = self._classifier_predict(line)
        if max(res[0]) > 0.1:
            return self.lb.inverse_transform(res)[0]
        else:
            return 'unknown'

    def classify2(self, line):
        res = self._classifier_predict(line)
        print('\n'.join(map(str, zip(self.lb.classes_, res[0]))))
        m = max(res[0])
        c = self.lb.inverse_transform(res)[0]
        if m > 0.05:
            return c
        elif m > 0.02:
            return 'probably ' + c
        else:
            return 'unknown ' + c + '? ' + str(m)

    def tag(self, line):
        X = pad_sequences([_embed(line)],
                          padding='post', maxlen=self.max_length)
        states = self.encoder.predict(X)

        target_seq = np.zeros((1, 1, self.max_length))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, self._embed('\t')[0]] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        decoded_sentence = []
        while True:
            output_tokens, h, c = self.decoder.predict([target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            decoded_sentence.append[sampled_token_index]

            # Exit condition: either hit max length
            # or find stop character.
            if len(decoded_sentence) == len(X[0]):
                break

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence
