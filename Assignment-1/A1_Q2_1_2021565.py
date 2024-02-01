import numpy as np
import pandas as pd
from math import log

from A1_Task1_2021529 import Tokenizer

class BiGramLM:
    def __init__(self, corpus = None,smoothing = None):
        self.tokenizer = Tokenizer()
        self.one_word_counts : dict[str,int] = dict()
        self.two_word_counts : dict[(str,str),int] = dict()
        self.smoothing : bool = smoothing
        self.corpus : list[str]= self.tokenizer.tokenize(corpus)
        self.corpus_size : int = len(self.corpus)
        self.conditional_log_probs = dict()

        for i in range(1,self.corpus_size):
            if self.corpus[i] not in self.one_word_counts:
                self.one_word_counts[self.corpus[i]] = 0
            self.one_word_counts[self.corpus[i]] += 1
            if (self.corpus[i-1],self.corpus[i]) not in self.two_word_counts:
                self.two_word_counts[(self.corpus[i-1],self.corpus[i])] = 0
            self.two_word_counts[(self.corpus[i-1],self.corpus[i])] += 1
        self.one_word_counts[self.corpus[0]] += 1
        for word1 in self.tokenizer.vocabulary:
            self.conditional_log_probs[word1] = []
            for word2 in self.tokenizer.vocabulary:
                self.conditional_log_probs[word1].append(
                        (word2,log(self.two_word_counts[(word1,word2)]
                            /self.one_word_counts[word1]))
                )
            self.conditional_log_probs[word1].sort(key=lambda x:x[1],reverse=True)
    
    def score(self, sentence:str) -> float:
        '''
        Score a sentence using the language model
        '''
        score = 0
        sentence = self.tokenizer.tokenize(sentence)
        for i in range(1,len(sentence)):
            if (sentence[i-1],sentence[i]) in self.two_word_counts:
                score += self.two_word_counts[(sentence[i-1],sentence[i])]
            else:
                score += 0
        return score
    
        