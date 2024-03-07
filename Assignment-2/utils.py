import os
import json
import torch
import gensim
import numpy as np
import transformers
from typing import Callable
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


class CustomDataset(Dataset):
    """
    A custom dataset class for dataset_1 and dataset_2.
    :attrs:
        - data: The data of the dataset (list of sentences).
        - targets: The targets of the dataset (list of labels).
        - encoder: The (already fitted) label-encoder for the targets.
        - vectorizor: The function to vectorize the data.
    """

    data: list[str]
    targets: list[str]
    encoder: LabelEncoder
    vectorizor: Callable[[str], np.ndarray]

    def __init__(self, data, targets, encoder, vectorizor):
        super(CustomDataset, self).__init__()
        self.data = data
        self.targets = targets
        self.encoder = encoder
        self.vectorizor = vectorizor

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor]:
        """
        Returns the data and target at the given index. The sentence is
        converted to a tensor of shape (len(sentence), 300) where each word
        is represented by its word-embedding. The target is converted to
        a label-encoded tensor.
        """
        data, target = self.data[index], self.targets[index]
        vector = torch.zeros(len(data), 300)
        for i, word in enumerate(data):
            try:
                vector[i] = torch.tensor(self.vectorizor(word))
            except KeyError:
                pass
        target = self.encoder.transform(target)
        return vector, torch.tensor(target)


def load_dataset(dataset: int, embeddings: str) -> tuple[CustomDataset]:
    """
    Loads the given dataset and returns the train, test, and validation sets
    as CustomDataset objects.
    :params:
        - dataset: The dataset to load. Must be 1 or 2.
        - embeddings: The word-embeddings to use. Must be 'Word2Vec', 'GloVe', or 'FastText' (case-insensitive).
    """

    embeddings_err = "Invalid embeddings. Must be 'Word2Vec', 'GloVe', or 'FastText'."
    dataset_err = "Invalid dataset number. Must be 1 or 2."
    embeddings = embeddings.casefold()

    assert dataset in [1, 2], dataset_err
    assert embeddings in ["word2vec", "glove", "fasttext"], embeddings_err

    base = r"Assignment-2/"
    train_path, test_path, val_path = [
        os.path.join(base, r"Datasets", rf"dataset_{dataset}/{x}.json")  for x in ["train", "test", "val"]
    ]

    with open(train_path) as train, open(test_path) as test, open(val_path) as val:
        train_data = json.load(train)
        test_data = json.load(test)
        val_data = json.load(val)

    print(f"Loading {embeddings} embeddings...")
    if embeddings == "word2vec":
        model = os.path.join(base, r"Embeddings/GoogleNews-vectors-negative300.bin.gz")
        embedding = gensim.models.KeyedVectors.load_word2vec_format(model, binary=True)
        vectorizor = embedding.get_vector
    elif embeddings == "glove":
        model = os.path.join(base, r"Embeddings/glove.42B.300d.bin.gz")
        embedding = gensim.models.KeyedVectors.load_word2vec_format(model, binary=True)
        vectorizor = embedding.get_vector
    else:
        model = os.path.join(base, r"Embeddings/cc.en.300.bin.gz")
        embedding = gensim.models.fasttext.load_facebook_model(model)
        vectorizor = lambda x: embedding.wv[x]

    LABELS = set()
    TRAIN_DATA, TRAIN_LABELS = [], []
    for data in train_data.values():
        TRAIN_DATA.append(data["text"].split())
        TRAIN_LABELS.append(data["labels"])
        LABELS.update(data["labels"])
    encoder = LabelEncoder()
    encoder.fit(sorted(LABELS))

    TEST_DATA, TEST_LABELS = [], []
    for data in test_data.values():
        TEST_DATA.append(data["text"].split())
        TEST_LABELS.append(data["labels"])

    VAL_DATA, VAL_LABELS = [], []
    for data in val_data.values():
        VAL_DATA.append(data["text"].split())
        VAL_LABELS.append(data["labels"])

    train_set = CustomDataset(TRAIN_DATA, TRAIN_LABELS, encoder, vectorizor)
    test_set = CustomDataset(TEST_DATA, TEST_LABELS, encoder, vectorizor)
    val_set = CustomDataset(VAL_DATA, VAL_LABELS, encoder, vectorizor)

    return train_set, test_set, val_set