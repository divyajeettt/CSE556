import numpy as np
import pandas as pd
from math import log

from A1_divyajeet import Tokenizer

class BiGramLM:
    def __init__(self, corpus = None,smoothing = None):
        tokenizer = Tokenizer()
        self.one_word_counts = dict()
        self.two_word_counts = dict()
        self.smoothing = smoothing
        self.corpus = tokenizer.tokenize(corpus)
        self.corpus_size = len(self.corpus)
        for i in range(1,self.corpus_size):
            if self.corpus[i] not in self.one_word_counts:
                self.one_word_counts[self.corpus[i]] = 0
            self.one_word_counts[self.corpus[i]] += 1
            if (self.corpus[i-1],self.corpus[i]) not in self.two_word_counts:
                self.two_word_counts[(self.corpus[i-1],self.corpus[i])] = 0
            self.two_word_counts[(self.corpus[i-1],self.corpus[i])] += 1
        self.one_word_counts[self.corpus[0]] += 1
        self.conditional_log_probs = dict()
        for i in 