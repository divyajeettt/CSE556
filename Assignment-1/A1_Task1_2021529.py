import re
import collections


class Tokenizer:
    """
    Implementation of a Tokenizer based on the Byte Pair Encoding (BPE) Algorithm.
    """

    corpus: list[str]
    vocabulary: collections.Counter
    merges: list[tuple[str, str]]
    pairs: collections.Counter
    tokens: frozenset[str]

    def __init__(self):
        self.merges = []
        self.pairs = collections.Counter()

    def learn_vocabulary(self, corpus: list[str], num_merges: int) -> None:
        self._fit(corpus)
        for _ in range(num_merges):
            self._make_pairs()
            pair = self.pairs.most_common(1)[0][0]
            self._merge_tokens(pair)
            self.merges.append(pair)
        self.tokens = frozenset(word for string in self.vocabulary.keys() for word in string.split())

    def tokenize(self, word: str) -> list[str]:
        tokens = " ".join(word)
        for pair in self.merges:
            tokens = self._merge_word(tokens, pair)
        return tokens.split()

    def _fit(self, corpus: list[str]) -> None:
        self.corpus = [" ".join(word) for word in corpus]
        self.vocabulary = collections.Counter(self.corpus)

    def _make_pairs(self):
        self.pairs.clear()
        for word, freq in self.vocabulary.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                self.pairs[symbols[i], symbols[i + 1]] += freq

    def _merge_tokens(self, pair: tuple[str, str]):
        previous = self.vocabulary
        self.vocabulary = collections.Counter()
        for word, freq in previous.items():
            merged = self._merge_word(word, pair)
            self.vocabulary[merged] = freq

    def _merge_word(self, word: str, pair: tuple[str, str]) -> str:
        bigram = re.escape(" ".join(pair))
        merged = "".join(pair)
        return re.sub(bigram, merged, word)