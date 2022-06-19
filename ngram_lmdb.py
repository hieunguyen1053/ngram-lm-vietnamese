import re

import lmdb

from regex_tokenize import datetime as datetime_pattern
from regex_tokenize import email as email_pattern
from regex_tokenize import number as number_pattern
from regex_tokenize import phone as phone_pattern
from regex_tokenize import url as url_pattern

PATTERNS = [
    datetime_pattern,
    email_pattern,
    number_pattern,
    phone_pattern,
    url_pattern
]
PATTERNS = r"(" + "|".join(PATTERNS) + ")"
PATTERNS = re.compile(PATTERNS, re.VERBOSE | re.UNICODE)

class NgramLMDB(object):
    def __init__(self, ngram_cased_path, ngram_uncased_path, alpha=0.1):
        self.env1 = lmdb.open(ngram_cased_path, readonly=True, lock=False, readahead=False, meminit=False)
        self.env2 = lmdb.open(ngram_uncased_path, readonly=True, lock=False, readahead=False, meminit=False)

        self.alpha = alpha

    def get_ngram_freq(self, ngram, cased=True):
        if cased:
            txn = self.env1.begin(write=False)
        else:
            txn = self.env2.begin(write=False)

        val = None
        key = '▁'.join(ngram)
        try:
            val = txn.get(key.encode())
        except lmdb.BadValsizeError:
            pass
        if val is not None:
            return float(val)
        return 0.0

    def get_ngram_prob(self, ngram, cased=True):
        if len(ngram) == 3:
            # P(w_3 | w_1, w_2)
            num = self.get_ngram_freq(ngram, cased)
            denom = self.get_ngram_freq(ngram[:-1], cased)
            if num != 0.0 and denom != 0.0:
                return num / denom

            # P(w_3 | w_1, w_2) = P(w_3 | w_2)
            num = self.get_ngram_freq(ngram[1:], cased)
            denom = self.get_ngram_freq(ngram[1:-1], cased)
            if num != 0.0 and denom != 0.0:
                return num / denom

            # P(w_3 | w_1, w_2) = C(w_3) / V
            num = self.get_ngram_freq([ngram[-1]], cased)
            V = self.get_vocab_size(cased)
            if num != 0.0:
                return num / V
            return (self.alpha + num) / (V + self.alpha * V)

        if len(ngram) == 2:
            # P(w_2 | w_1)
            num = self.get_ngram_freq(ngram, cased)
            denom = self.get_ngram_freq(ngram[:-1], cased)
            if num != 0.0 and denom != 0.0:
                return num / denom

            # P(w_2 | w_1) = C(w_2) / V
            num = self.get_ngram_freq([ngram[-1]], cased)
            V = self.get_vocab_size(cased)
            if num != 0.0:
                return num / V
            return (self.alpha + num) / (V + self.alpha * V)

        if len(ngram) == 1:
            num = self.get_ngram_freq([ngram[-1]], cased)
            V = self.get_vocab_size(cased)
            if num != 0.0:
                return num / V
            return (self.alpha + num) / (V + self.alpha * V)
        raise ValueError('ngram is empty')

    def get_vocab_size(self, cased=True):
        return 5407254479
        if cased:
            txn = self.env1.begin(write=False)
        else:
            txn = self.env2.begin(write=False)

        val = txn.get('▁V▁'.encode())
        if val is not None:
            return int(val)
        return 0

NGRAM_LMDB = NgramLMDB('ngram-cased', 'ngram-uncased')

class Ngram(object):
    def __init__(self, words):
        self._words = words
        self.cased_words = []
        self.uncased_words = []

        self._process()

    def _process(self):
        self.cased_words = []
        self.uncased_words = []
        for word in self._words:
            flag = False
            m = re.match(PATTERNS, word)
            if m is not None:
                for k, v in m.groupdict().items():
                    if v is not None:
                        word = k.upper()
                        flag = True
                        break
            if flag:
                self.cased_words.append(word)
                self.uncased_words.append(word)
            else:
                self.cased_words.append(word)
                self.uncased_words.append(word.lower())

    @property
    def probability(self, ngram_lmdb=NGRAM_LMDB):
        prob = ngram_lmdb.get_ngram_prob(self.cased_words, cased=True)
        if prob == 0.0:
            prob = ngram_lmdb.get_ngram_prob(self.uncased_words, cased=False) * 0.95
        return prob

    @property
    def frequency(self, ngram_lmdb=NGRAM_LMDB):
        freq = ngram_lmdb.get_ngram_freq(self.cased_words, cased=True)
        if freq == 0.0:
            freq = ngram_lmdb.get_ngram_freq(self.uncased_words, cased=False) * 0.95
        return freq

    def __str__(self):
        return '{} {} {}'.format(self.w1, self.w2, self.w3)