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

    def learn_vocabulary(self, corpus: list[str], num_merges: int, add_eos: bool|None = False) -> None:
        self._fit(corpus, add_eos=add_eos)
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

    def _fit(self, corpus: list[str], add_eos: bool) -> None:
        self.corpus = [" ".join(word) for word in corpus]
        if add_eos:
            self.corpus = [word + "$" for word in self.corpus]
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


def main():
    tokenizer = Tokenizer()
    with open("Dataset/corpus.txt", "r") as file:
        corpus = file.read()

    N_MERGES = input("Input the number of merges: ")
    try:
        N_MERGES = int(N_MERGES)
    except ValueError:
        print("Invalid input. Please enter an integer.")
        return
    tokenizer.learn_vocabulary(corpus.split(), N_MERGES, add_eos=True)

    # Save the merges
    with open("Task-1-Results/merge_rules.txt", "w") as file:
        for merge in tokenizer.merges:
            file.write(f"{merge[0]},{merge[1]}\n")

    # Save the tokens
    with open("Task-1-Results/tokens.txt", "w") as file:
        for token in sorted(tokenizer.tokens, key=len):
            file.write(f"{token}\n")

    # Save the tokenized samples
    with open("Task-1-Results/tokenized_samples.txt", "w") as file:
        for line in corpus.splitlines():
            tokenized = tokenizer.tokenize(line)
            file.write(",".join(tokenized))
            file.write("\n")


if __name__ == "__main__":
    main()