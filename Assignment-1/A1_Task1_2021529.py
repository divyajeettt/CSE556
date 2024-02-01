import re
import collections


class Tokenizer:
    """
    Implementation of a Tokenizer based on the Byte Pair Encoding (BPE) Algorithm.
    The learnt vocabulary for a Tokenizer object is stored in the attr vocabulary.
    """

    vocabulary: dict[str, int]

    def __init__(self):
        pass

    def learn_vocabulary(self, corpus: list[str], num_merges: int):
        pass

    def tokenize(self, word: str) -> list[str]:
        pass