import numpy as np
import pandas as pd
from math import exp
import math
from typing import Union, Tuple
from collections import Counter

from A1_Task1_2021529 import Tokenizer

def log(x):
    '''
    Prevents taking log of zeroes
    '''
    if x <= 0:
        return -1e9
    return math.log(x)


class BiGramLM:
    def __init__(self, corpus :str,smoothing : Union[str,None] = None,num_merges : int = 400):
        print('Tokenizing')
        self.tokenizer = Tokenizer()
        self.tokenizer.learn_vocabulary(corpus.split(),num_merges=num_merges)
        self.one_word_counts : Counter[str] = Counter()
        self.two_word_counts : Counter[Tuple[str,str]] = Counter()
        self.smoothing = smoothing
        self.corpus : list[str] = self.tokenizer.tokenize(corpus.split())
        self.corpus_size : int = len(self.corpus)
        self.conditional_log_probs = dict()
        self.smoothing = smoothing
        print('Preprocessing')
        self.preprocess()
        print('Populating table')
        self.populate_table()
        print('Calculating probabilities')
        self.calculate_probs()

    def preprocess(self):
        '''
        If you need to add any other stuff before populating the table
        '''
        pass

    def getfreq(self, word1 : str, word2 : Union[None,str] = None) -> int:
        '''
        Get the frequency of a word or a pair of words
        '''
        if word2 is None:
            if word1 not in self.one_word_counts:
                return 1e-10
            return self.one_word_counts[word1]
        if (word1,word2) not in self.two_word_counts:
            return 0
        return self.two_word_counts[(word1,word2)]

    def score(self, sentence:str) -> float:
        '''
        Score the perplexity of a sentence using the language model
        '''
        score = 0
        sentence = self.tokenizer.tokenize(sentence)
        for i in range(1,len(sentence)):
            score += self.conditional_prob(sentence[i-1],sentence[i])
        score /= -len(sentence)
        return score
    
    def populate_table(self):
        self.one_word_counts = Counter(self.corpus)
        pairs = [(self.corpus[i],self.corpus[i+1]) for i in range(len(self.corpus)-1)]
        self.two_word_counts = Counter(pairs)

    def calculate_probs(self):
        for word1 in self.tokenizer.tokens:
            self.conditional_log_probs[word1] = []
            for word2 in self.tokenizer.tokens:
                self.conditional_log_probs[word1].append(
                        (word2,self.conditional_prob(word1,word2))
                )
            self.conditional_log_probs[word1].sort(key=lambda x:x[1],reverse=True)

    def conditional_prob(self, word1 : str, word2 : str) -> float:
        '''
        Calculate the conditional log probability of word2 given word1
        '''
        if self.smoothing is None:
            prob = self.getfreq(word1,word2)/self.getfreq(word1)
        elif self.smoothing == 'laplace':
            prob = (self.getfreq(word1,word2)+1)/(self.getfreq(word1)+len(self.tokenizer.tokens))
        elif self.smoothing == 'kneser-ney':
            #Hardcoded
            discount = 0.5
            lamda = discount/self.getfreq(word1)
            pcontinuation = self.getfreq(word1)/self.corpus_size
            prob = max(self.getfreq(word1,word2)-discount,0)/self.getfreq(word1) + lamda*pcontinuation*self.getfreq(word2)
        return log(prob)


    def sample(self, prior : str):
        '''
        Sample a word from the conditional distribution given a prior
        '''
        probabilities = [exp(x[1]) for x in self.conditional_log_probs[prior]]
        words = [x[0] for x in self.conditional_log_probs[prior]]
        #Normalize probabilities
        normed_probabilities = [x/sum(probabilities) for x in probabilities]
        return np.random.choice(words,p=normed_probabilities)

    def generate(self, num_words: int, start : Union[None,str] = None) -> str:
        '''
        Generate a sentence from the language model
        '''
        if start is None:
            start = np.random.choice(list(self.tokenizer.tokens))
        sentence = start
        last_token = self.tokenizer.tokenize(start)[-1]
        for i in range(num_words):
            sentence += ' '
            last_token = self.sample(last_token)
            sentence += last_token
        return sentence