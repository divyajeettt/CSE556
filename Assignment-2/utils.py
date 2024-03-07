import os
import json
import wandb
import torch
import gensim
from typing import Any
import sklearn.metrics as metrics
from sklearn.preprocessing import LabelEncoder


WordEmbedding = gensim.models.keyedvectors.KeyedVectors | gensim.models.fasttext.FastTextKeyedVectors


class CustomDataset(torch.utils.data.Dataset):
    """
    A custom dataset class for dataset_1 and dataset_2.
    :attrs:
        - data: The data of the dataset. Each element is a list of words.
        - labels: The labels of the dataset. Each element is a list of labels
            for each word of the sentence.
        - encoder: The (already fitted) label-encoder for the labels.
        - embedding: The word-embedding to use for the dataset. It must have
            a key_to_index attribute attribute (to convert words to indices).

    The dataset preprocesses the input data and labels. 'data' is stored as
    a Tensor of word-indices, and 'labels' is stored as a Tensor of label-indices.
    """

    data: torch.Tensor
    labels: torch.Tensor
    encoder: LabelEncoder
    embedding: WordEmbedding

    def __init__(self, data: list[list[str]], labels: list[list[str]], encoder, embedding):
        super(CustomDataset, self).__init__()
        self.data = data
        self.labels = labels
        self.encoder = encoder
        self.embedding = embedding
        self._encode()
        self._pad()

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor]:
        """
        Returns the data and target at the given index. Converts each
        sentence into a tensor of word indices using the chosen word-embedding.
        """
        return self.data[index], self.labels[index]

    def _encode(self) -> None:
        """
        Converts all sentences to their corresponding word-indices and
        encodes the labels using the label-encoder.
        """
        unk = len(self.embedding.key_to_index) - 1
        self.data = [
            torch.tensor([self.embedding.key_to_index.get(word, unk) for word in sentence])
            for sentence in self.data
        ]
        self.labels = [
            torch.LongTensor(self.encoder.transform(label)) for label in self.labels
        ]

    def _pad(self) -> None:
        """
        Pads all sentences to the maximum length. Currently, the padding
        value for the labels is set to "O", which may not be ideal.
        """
        padding_value = self.encoder.transform(["O"])[0]
        self.data = torch.nn.utils.rnn.pad_sequence(self.data, batch_first=True)
        self.labels = torch.nn.utils.rnn.pad_sequence(
            self.labels, batch_first=True, padding_value=padding_value
        )


class RNN(torch.nn.Module):
    """
    A class for the vanilla RNN model. The model can be used for both NER
    and ATE datasets. The model design is simply:
        - An embedding layer
        - 'num_layers' number of RNN layers
        - A fully connected layer with 'output_size' number of units
        - A LogSoftmax layer for the output log probabilities

    The embedding matrix is the word-embedding matrix given by the chosen
    embedding model and must have an additional row for the <UNK> token
    (will be added to your hyperparameters automatically through the pipeline).
    """

    input_size: int
    hidden_size: int
    num_layers: int
    output_size: int
    embedding_matrix: torch.Tensor

    def __init__(self, input_size, hidden_size, num_layers, output_size, embedding_matrix):
        super(RNN, self).__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(embedding_matrix)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.LogSoftmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the model. Returns the log probabilities of the
        output classes for each word for each sentence in the batch. The hidden
        states are handled internally by torch.nn.RNN.
        """
        x = self.embedding(x)
        hidden = torch.zeros(self.num_layers, x.shape[0], self.hidden_size)
        output, _ = self.rnn(x, hidden)
        output = self.fc(output)
        return self.softmax(output)


class LSTM(torch.nn.Module):
    pass


class GRU(torch.nn.Module):
    pass


class BiLSTM_CRF(torch.nn.Module):
    pass


def load_dataset(dataset: str, embedding: str) -> tuple[CustomDataset]:
    """
    Loads the given dataset and returns the train, test, and validation sets
    with the corresponding labels. It is expected that this function will be
    called within the pipeline after the configurations have been checked.
    :params:
        - dataset: The dataset to load. Must be 'NER' or 'ATE' (case-insensitive).
        - embedding: The word-embedding to use. Must be 'Word2Vec', 'GloVe', or 'FastText' (case-insensitive).
    """

    path = r"Assignment-2/Datasets/preprocessed/"
    train_path, test_path, val_path = [
        os.path.join(path, dataset, f"{dataset}_{x}.json") for x in ["train", "test", "val"]
    ]

    with open(train_path) as train, open(test_path) as test, open(val_path) as val:
        train_data = json.load(train)
        test_data = json.load(test)
        val_data = json.load(val)

    print("Loading Word Embeddings...")
    if embedding == "word2vec":
        model = "Assignment-2/Embeddings/GoogleNews-vectors-negative300.bin.gz"
        embedding_model = gensim.models.KeyedVectors.load_word2vec_format(model, binary=True)
    elif embedding == "glove":
        model = "Assignment-2/Embeddings/glove.42B.300d.bin.gz"
        embedding_model = gensim.models.KeyedVectors.load_word2vec_format(model, binary=True)
    else:
        model = "Assignment-2/Embeddings/cc.en.300.bin.gz"
        embedding_model = gensim.models.fasttext.load_facebook_model(model).wv

    print("Preprocessing Data...")
    LABELS = set()
    for data in train_data.values():
        LABELS.update(data["labels"])
    encoder = LabelEncoder()
    encoder.fit(sorted(LABELS))

    DATA = [[], []], [[], []], [[], []]
    for i, dataset in enumerate([train_data, test_data, val_data]):
        for data in dataset.values():
            DATA[i][0].append(data["text"].split())
            DATA[i][1].append(data["labels"])
    (TRAIN_DATA, TRAIN_LABELS), (TEST_DATA, TEST_LABELS), (VAL_DATA, VAL_LABELS) = DATA

    train_set = CustomDataset(TRAIN_DATA, TRAIN_LABELS, encoder, embedding_model)
    test_set = CustomDataset(TEST_DATA, TEST_LABELS, encoder, embedding_model)
    val_set = CustomDataset(VAL_DATA, VAL_LABELS, encoder, embedding_model)

    return train_set, test_set, val_set


def get_embedding_matrix(embedding: WordEmbedding) -> torch.Tensor:
    """
    Given an emedding model, returns the corresponding embedding matrix. The
    embedding matrix is extended by one row to handle OOV words. This function
    handles the <UNK> token, i.e. OOV words. Currently, OOV words are directly
    set to 0s, which is not ideal.
    Note: The embedding model is also modified to include the <UNK> token.
    """
    embedding_matrix = torch.FloatTensor(embedding.vectors)
    embedding_matrix = torch.cat([embedding_matrix, torch.zeros(1, embedding_matrix.shape[1])])
    embedding.key_to_index["<UNK>"] = len(embedding.key_to_index)
    embedding.index_to_key.append("<UNK>")
    return embedding_matrix


def check_configs(configs: dict[str, Any]) -> None:
    """
    Checks the given configurations for any invalid values. The configurations
    expected are listed in help(pipeline).
    """

    dataset_err = "Invalid dataset. Must be 'NER' or 'ATE'."
    condfigs["dataset"] = configs.get("dataset", "").upper()
    assert configs["dataset"] in ["NER", "ATE"], dataset_err

    embedding_err = "Invalid embedding. Must be 'Word2Vec', 'GloVe', or 'FastText'."
    configs["embedding"] = configs.get("embedding", "").casefold()
    assert configs["embedding"] in ["word2vec", "glove", "fasttext"], embedding_err

    model_err = "Invalid model. Must be 'RNN', 'LSTM', 'GRU', or 'BiLSTM-CRF'."
    configs["model"] = configs.get("model", "").casefold()
    assert configs["model"] in ["rnn", "lstm", "gru", "bilstm-crf"], model_err

    batch_size_err = "Batch size must be a positive integer."
    configs["batch_size"] = configs.get("batch_size", 0)
    assert type(configs["batch_size"]) is int and configs["batch_size"] > 0, batch_size_err

    epochs_err = "Number of epochs must be a positive integer."
    configs["epochs"] = configs.get("epochs", 0)
    assert type(configs["epochs"]) is int and configs["epochs"] > 0, epochs_err

    lr_err = "Learning rate must be a float in (0.0, 1.0]."
    configs["lr"] = configs.get("lr", 0.0)
    assert type(configs["lr"]) is float and 0.0 < configs["lr"] <= 1.0, lr_err

    criterion_err = "Invalid criterion. Must be 'NLLLoss', 'CrossEntropy', or 'CRF'."
    configs["criterion"] = configs.get("criterion", "").casefold()
    assert configs["criterion"] in ["nllloss", "crossentropy", "crf"], criterion_err

    optimizer_err = "Invalid optimizer. Must be 'Adam', 'Adagrad', or 'SGD'."
    configs["optimizer"] = configs.get("optimizer", "").casefold()
    assert configs["optimizer"] in ["adam", "adagrad", "sgd"], optimizer_err

    hyperparams_err = "Hyperparameters must be a dictionary."
    configs["hyperparams"] = configs.get("hyperparams", {})
    assert type(configs["hyperparams"]) is dict, hyperparams_err

    configs["verbose"] = configs.get("verbose", False)


def train(
        model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, epochs: int, verbose: bool
    ) -> None:
    """
    Trains the given model using the given configurations. It is expected
    that this function is run through the pipeline after the configurations
    have been checked.
    """

    wandb.watch(model, criterion, log="all", log_freq=5)

    num_classes = len(train_loader.dataset.encoder.classes_)
    best_val_loss = float("inf")
    patience, counter = 3, 0

    for epochs in range(1, epochs+1):
        model.train()
        train_loss, train_f1 = run_epoch(model, train_loader, optimizer, criterion, num_classes, evaluate=False)
        with torch.no_grad():
            model.eval()
            val_loss, val_f1 = run_epoch(model, val_loader, optimizer, criterion, num_classes, evaluate=True)
        if verbose:
            epoch = f"[Epoch: {epoch}/{epochs}]"
            train = f"Loss: {train_loss:.5f}, F1-Score: {train_f1:.5f}"
            val = f"Loss: {val_loss:.5f}, F1-Score: {val_f1:.5f}"
            print(f"{epoch} Train: {train}, Validation: {val}", end="\r")

        if val_loss < best_val_loss:
            best_val_loss, counter = val_loss, 0
        else:
            if (counter := counter + 1) >= patience:
                print(f"\nEarly Stopping at Epoch {epoch}.")
                break

        wandb.log({
            "Train Loss": train_loss,
            "Train F1-Score": train_f1,
            "Validation Loss": val_loss,
            "Validation F1-Score": val_f1
        }, step=epoch)

    if verbose:
        print("\nTraining Complete.")


def evaluate(
        model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, criterion: torch.nn.Module,
    ) -> tuple[float]:
    """
    Evaluates the given model using the given configurations. It is expected
    that this function is run through the pipeline after the configurations
    have been checked. Returns the loss and f1-score for the given data (test) set.
    """

    model.eval()
    num_classes = len(dataloader.dataset.encoder.classes_)
    with torch.no_grad():
        return run_epoch(model, dataloader, None, criterion, num_classes, evaluate=True)


def run_epoch(
        model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module, num_classes: int, evaluate: bool
    ) -> tuple[float]:
    """
    Runs a single epoch of training or validation. The model is trained
    if evaluate is False, and evaluated if evaluate is True. Returns the
    loss and f1-score for the epoch.
    """

    epoch_loss, true, predicted = 0, [], []
    for data, labels in dataloader:
        output = model(data).permute(0, 2, 1)
        mask = (data != 0)
        labels = labels * mask
        output = output * mask.unsqueeze(1).repeat(1, num_classes, 1).float()
        epoch_loss += (loss := criterion(output, labels)).item()
        true.extend(labels.flatten().tolist())
        predicted.extend(output.argmax(dim=1).flatten().tolist())
        if not evaluate:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    f1 = metrics.f1_score(true, predicted, average="macro")
    return epoch_loss/len(dataloader), f1


def pipeline(configs: dict[str, Any]) -> None:
    """
    The pipeline works in 3 steps:
        - Performing configuration checks
        - Training and saving the model
        - Evaluating the model

    Trains the required model using the given configurations. To train any model for this
    assignment, pass the configurations to this function. The pipeline will train, save,
    and evaluate the model.
    :param configs: A dictionary containing the configurations for training a model.
    The dictionary must define the following keys:
        - model: The model to train. Must be 'RNN', 'LSTM', 'GRU', or 'BiLSTM-CRF'.
        - dataset: The dataset to use. Must be 'NER' or 'ATE'.
        - embedding: The word-embedding to use. Must be 'Word2Vec', 'GloVe', or 'FastText'.
        - batch_size: The batch size for the dataloaders.
        - epochs: The number of epochs to train the model.
        - lr: The learning rate for the optimizer.
        - criterion: The loss function to use. Must be 'NLLLoss', 'CrossEntropy', or 'CRF'.
        - optimizer: The optimizer to use. Must be 'Adam', 'Adagrad', or 'SGD'.
        - hyperparams: A dictionary containing the hyperparameters for the model. This
            must match the hyperparams expected by the model.
        - verbose: Whether to print the training progress or not. Default is False.
    Note: All values are case-insensitive.
    """

    check_configs(configs)

    train_set, test_set, val_set = load_dataset(configs["dataset"], configs["embedding"])
    embedding_matrix = get_embedding_matrix(train_set.embedding)

    hyperparams = configs.get("hyperparams", {})
    hyperparams["embedding_matrix"] = embedding_matrix
    model_name = configs["model"]
    model = {"rnn": RNN, "lstm": LSTM, "gru": GRU, "bilstm-crf": BiLSTM_CRF}[model_name](**hyperparams)

    batch_size = configs["batch_size"]
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

    criterion = configs["criterion"]
    criterion = {"nllloss": nn.NLLLoss, "crossentropy": nn.CrossEntropyLoss, "crf": None}[criterion]()

    optimizer = configs["optimizer"]
    optimizer = {"adam": torch.optim.Adam, "adagrad": torch.optim.Adagrad, "sgd": torch.optim.SGD}[optimizer]
    optimizer = optimizer(model.parameters(), lr=lr)

    task = "t1" if configs["dataset"] == "NER" else "t2"
    run = f"{task}_{model_name}_{configs['embedding']}"

    with wandb.init(project=run, config=hyperparams):
        train(model, train_loader, val_loader, optimizer, criterion, configs["epochs"], configs["verbose"])
        torch.save(model.state_dict(), model_path)
        loss, f1 = evaluate(model, test_loader, criterion)
        print(f"Test Loss: {loss:.5f}, Test F1-Score: {f1:.5f}")
        model_path = fr"Assignment-2/Models/{run}.pt"