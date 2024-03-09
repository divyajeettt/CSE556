import os
import json
import tqdm
import torch
import gensim
import pickle
import seaborn as sns
from typing import Any
import matplotlib.pyplot as plt
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
    """
    A class for the LSTM model. The model can be used for both NER and
    ATE datasets. The model design is simply:
        - An embedding layer
        - 'num_layers' number of LSTM layers
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
        super(LSTM, self).__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(embedding_matrix)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.LogSoftmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the model. Returns the log probabilities of the
        output classes for each word for each sentence in the batch. The hidden
        states are handled internally by torch.nn.LSTM.
        """
        x = self.embedding(x)
        hidden = (
            torch.zeros(self.num_layers, x.shape[0], self.hidden_size),
            torch.zeros(self.num_layers, x.shape[0], self.hidden_size)
        )
        output, _ = self.lstm(x, hidden)
        output = self.fc(output)
        return self.softmax(output)


class GRU(torch.nn.Module):
    """
    A class for the GRU model. The model can be used for both NER and ATE
    datasets. The model design is simply:
        - An embedding layer
        - 'num_layers' number of GRU layers
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
        super(GRU, self).__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(embedding_matrix)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.LogSoftmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the model. Returns the log probabilities of the
        output classes for each word for each sentence in the batch. The hidden
        states are handled internally by torch.nn.GRU.
        """
        x = self.embedding(x)
        hidden = torch.zeros(self.num_layers, x.shape[0], self.hidden_size)
        output, _ = self.gru(x, hidden)
        output = self.fc(output)
        return self.softmax(output)

class CRF(torch.nn.Module):
    def __init__(self, num_tags):
        super(CRF, self).__init__()
        self.num_tags = num_tags + 2
        self.start = self.num_tags-2
        self.end = self.start+1
        self.transitions = torch.nn.Parameter(torch.randn(self.num_tags, self.num_tags))
    
    def forward_score(self,features):
        scores = torch.ones(features.shape[0], self.num_tags) * -6969
        scores[:,self.start] = 0
        for i in range(features.shape[1]):
            feat = features[:,i]
            score = scores.unsqueeze(1) + feat.unsqueeze(2) + self.transitions.unsqueeze(0)
            scores = torch.logsumexp(score, dim=-1)
        scores = scores + self.transitions[self.end]
        return torch.logsumexp(scores, dim=-1)
    
    def score_sentence(self,features,tags):
        scores = features.gather(2, tags.unsqueeze(2)).squeeze(2)
        start = torch.ones(features.shape[0],1,dtype=torch.long) * self.start
        tags = torch.cat([start,tags],dim=1)
        trans_scores = self.transitions[tags[:,:-1],tags[:,1:]].sum(dim=1)
        last_tags = torch.gather(tags,1,torch.ones(tags.shape,dtype=torch.long) * tags.shape[1]-1)
        last_scores = self.transitions[self.end,last_tags]
        return (trans_scores + scores).sum(dim=1) + last_scores
    
    def viterbi_decode(self,features):
        scores = torch.ones(features.shape[0], self.num_tags) * -6969
        scores[:,self.start] = 0
        paths = []
        for i in range(features.shape[1]):
            feat = features[:,i]
            score = scores.unsqueeze(1) + feat.unsqueeze(2) + self.transitions.unsqueeze(0)
            scores, idx = score.max(dim=-1)
            paths.append(idx)
        scores = scores + self.transitions[self.end]
        scores, idx = scores.max(dim=-1)
        best_path = [idx]
        for path in reversed(paths):
            idx = path.gather(1,idx)
            best_path.append(idx)
        best_path = torch.cat(list(reversed(best_path)),dim=1)
        return scores, best_path
    
    def forward(self,features):
        return self.viterbi_decode(features)

    def loss(self,features,tags):
        forward_score = self.forward_score(features)
        gold_score = self.score_sentence(features,tags.long())
        return (forward_score - gold_score).mean()

class BiLSTM_CRF(torch.nn.Module):
    """
    A class for the GRU model. The model can be used for both NER and ATE
    datasets. The model design is simply:
        - An embedding layer
        - TO-BE-IMPLEMENTED

    The embedding matrix is the word-embedding matrix given by the chosen
    embedding model and must have an additional row for the <UNK> token
    (will be added to your hyperparameters automatically through the pipeline).
    """

    # parameter: type
    embedding_matrix: torch.Tensor

    # [ideally, match the signature of the other models - RNN, LSTM, GRU]
    # must at least have the embedding matrix as a parameter (due to data preprocessing)
    def __init__(self, embedding_matrix, /, *args):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(embedding_matrix)
        # TO-BE-IMPLEMENTED

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the model. Must return the log probabilities of the
        output classes for each word for each sentence in the batch (assuming
        BiLSTM-CRF works in the same way).
        x.shape = (batch_size, max_sentence_length)
        out.shape = (batch_size, max_sentence_length, num_classes)
        """
        # TO-BE-IMPLEMENTED
        pass


def load_dataset(dataset: str, embedding: str, verbose: bool) -> tuple[CustomDataset]:
    """
    Loads the given dataset and returns the train, test, and validation sets
    with the corresponding labels. It is expected that this function will be
    called within the pipeline after the configurations have been checked.
    :params:
        - dataset: The dataset to load. Must be 'NER' or 'ATE' (case-insensitive).
        - embedding: The word-embedding to use. Must be 'Word2Vec', 'GloVe', or 'FastText' (case-insensitive).
        - verbose: Whether to print the progress of the function or not.
    """

    path = r"Datasets/preprocessed/"
    train_path, test_path, val_path = [
        os.path.join(path, dataset, f"{dataset}_{x}.json") for x in ["train", "test", "val"]
    ]

    with open(train_path) as train, open(test_path) as test, open(val_path) as val:
        train_data = json.load(train)
        test_data = json.load(test)
        val_data = json.load(val)

    if verbose:
        print("Loading Word Embeddings...")

    if embedding == "word2vec":
        model = "Embeddings/GoogleNews-vectors-negative300.bin.gz"
        embedding_model = gensim.models.KeyedVectors.load_word2vec_format(model, binary=True)
    elif embedding == "glove":
        model = "Embeddings/glove.42B.300d.bin.gz"
        embedding_model = gensim.models.KeyedVectors.load_word2vec_format(model, binary=True)
    else:
        model = "Embeddings/cc.en.300.bin.gz"
        embedding_model = gensim.models.fasttext.load_facebook_model(model).wv

    if verbose:
        print("Preprocessing Data...")

    labels = set()
    for data in train_data.values():
        labels.update(data["labels"])
    encoder = LabelEncoder()
    encoder.fit(sorted(labels))

    data = [[], []], [[], []], [[], []]
    for i, dataset in enumerate([train_data, test_data, val_data]):
        for entry in dataset.values():
            data[i][0].append(entry["text"].split())
            data[i][1].append(entry["labels"])
    (train_data, train_labels), (test_data, test_labels), (val_data, val_labels) = data

    train_set = CustomDataset(train_data, train_labels, encoder, embedding_model)
    test_set = CustomDataset(test_data, test_labels, encoder, embedding_model)
    val_set = CustomDataset(val_data, val_labels, encoder, embedding_model)

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


def check_config(config: dict[str, Any]) -> None:
    """
    Checks the given configurations for any invalid values. The configurations
    expected are listed in help(pipeline).
    """

    dataset_err = "Invalid dataset. Must be 'NER' or 'ATE'."
    config["dataset"] = config.get("dataset", "").upper()
    assert config["dataset"] in ["NER", "ATE"], dataset_err

    embedding_err = "Invalid embedding. Must be 'Word2Vec', 'GloVe', or 'FastText'."
    config["embedding"] = config.get("embedding", "").casefold()
    assert config["embedding"] in ["word2vec", "glove", "fasttext"], embedding_err

    model_err = "Invalid model. Must be 'RNN', 'LSTM', 'GRU', or 'BiLSTM-CRF'."
    config["model"] = config.get("model", "").casefold()
    assert config["model"] in ["rnn", "lstm", "gru", "bilstm-crf"], model_err

    batch_size_err = "Batch size must be a positive integer."
    config["batch_size"] = config.get("batch_size", 0)
    assert type(config["batch_size"]) is int and config["batch_size"] > 0, batch_size_err

    epochs_err = "Number of epochs must be a positive integer."
    config["epochs"] = config.get("epochs", 0)
    assert type(config["epochs"]) is int and config["epochs"] > 0, epochs_err

    lr_err = "Learning rate must be a float in (0.0, 1.0]."
    config["lr"] = config.get("lr", 0.0)
    assert type(config["lr"]) is float and 0.0 < config["lr"] <= 1.0, lr_err

    criterion_err = "Invalid criterion. Must be 'NLLLoss', 'CrossEntropy', or 'CRF'."
    config["criterion"] = config.get("criterion", "").casefold()
    assert config["criterion"] in ["nllloss", "crossentropy", "crf"], criterion_err

    optimizer_err = "Invalid optimizer. Must be 'Adam', 'Adagrad', or 'SGD'."
    config["optimizer"] = config.get("optimizer", "").casefold()
    assert config["optimizer"] in ["adam", "adagrad", "sgd"], optimizer_err

    hyperparams_err = "Hyperparameters must be a dictionary."
    config["hyperparams"] = config.get("hyperparams", {})
    assert type(config["hyperparams"]) is dict, hyperparams_err

    config["device"] = config.get("device", "cpu")
    try:
        config["device"] = torch.device(config["device"])
    except Exception as err:
        print("Invalid device. ERROR:", err)
        print("Using default device: 'cpu'.")
        config["device"] = torch.device("cpu")

    config["early_stopping_patience"] = config.get("early_stopping_patience", float("inf"))
    config["verbose"] = config.get("verbose", False)


def train(
        model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, epochs: int, patience: int, verbose: bool
    ) -> None:
    """
    Trains the given model using the given configurations. It is expected
    that this function is run through the pipeline after the configurations
    have been checked.
    Ideally, model should be RNN|LSTM|GRU|BiLSTM_CRF.
    """

    num_classes = len(train_loader.dataset.encoder.classes_)
    best_val_loss, counter = float("inf"), 0

    device = next(model.parameters()).device
    model.LOSSES = torch.zeros(epochs, 2)
    model.F1_SCORES = torch.zeros(epochs, 2)

    progress_bar = tqdm.tqdm(range(epochs), bar_format=r"{l_bar}{bar:15}{r_bar}")
    for epoch in progress_bar if verbose else range(epochs):
        model.train()
        train_loss, train_true, train_predicted = 0, [], []
        for data, labels in train_loader:
            output = model(data).permute(0, 2, 1)
            mask = (data != 0)
            labels = labels * mask
            output = output * mask.unsqueeze(1).repeat(1, num_classes, 1).float()
            train_loss += (loss := criterion(output, labels)).item()
            train_true.extend(labels.flatten().tolist())
            train_predicted.extend(output.argmax(dim=1).flatten().tolist())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        with torch.no_grad():
            val_loss, val_true, val_predicted = 0, [], []
            for data, labels in val_loader:
                output = model(data).permute(0, 2, 1)
                mask = (data != 0)
                labels = labels * mask
                output = output * mask.unsqueeze(1).repeat(1, num_classes, 1).float()
                val_loss += (loss := criterion(output, labels)).item()
                val_true.extend(labels.flatten().tolist())
                val_predicted.extend(output.argmax(dim=1).flatten().tolist())

        model.LOSSES[epoch, 0] = train_loss / len(train_loader)
        model.LOSSES[epoch, 1] = val_loss / len(val_loader)
        model.F1_SCORES[epoch, 0] = metrics.f1_score(train_true, train_predicted, average="macro")
        model.F1_SCORES[epoch, 1] = metrics.f1_score(val_true, val_predicted, average="macro")

        if verbose:
            train = f"Loss: {model.LOSSES[epoch, 0]:.5f}, F1-Score: {model.F1_SCORES[epoch, 0]:.5f}"
            val = f"Loss: {model.LOSSES[epoch, 1]:.5f}, F1-Score: {model.F1_SCORES[epoch, 1]:.5f}"
            progress_bar.set_postfix_str(f"[Train: {train}], [Validation: {val}]")

        if val_loss < best_val_loss:
            best_val_loss, counter = val_loss, 0
        else:
            if (counter := counter + 1) >= patience:
                print(f"\nEarly Stopping at Epoch {epoch}.")
                model.LOSSES = model.LOSSES[:epoch + 1]
                model.F1_SCORES = model.F1_SCORES[:epoch + 1]
                break


def evaluate(
        model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, criterion: torch.nn.Module
    ) -> dict[str, float|torch.Tensor]:
    """
    Evaluates the given model using the given configurations. It is expected
    that this function is run through the pipeline after the configurations
    have been checked. Returns the evaluation metrics for the given test set.
    Ideally, model should be RNN|LSTM|GRU|BiLSTM_CRF.
    """

    num_classes = len(dataloader.dataset.encoder.classes_)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        test_loss, test_true, test_predicted = 0, [], []
        for data, labels in dataloader:
            output = model(data).permute(0, 2, 1)
            mask = (data != 0)
            labels = labels * mask
            output = output * mask.unsqueeze(1).repeat(1, num_classes, 1).float()
            test_loss += (loss := criterion(output, labels)).item()
            test_true.extend(labels.flatten().tolist())
            test_predicted.extend(output.argmax(dim=1).flatten().tolist())

    test_details = {
        "loss": test_loss / len(dataloader),
        "accuracy": metrics.accuracy_score(test_true, test_predicted),
        "precision": metrics.precision_score(test_true, test_predicted, average="macro"),
        "recall": metrics.recall_score(test_true, test_predicted, average="macro"),
        "f1": metrics.f1_score(test_true, test_predicted, average="macro"),
        "cf": metrics.confusion_matrix(test_true, test_predicted, normalize="true")
    }

    print(f"Test Loss: {test_details['loss']:.5f}")
    print(
        f"Accuracy: {test_details['accuracy']:.5f}, Precision: {test_details['precision']:.5f}"
        f", Recall: {test_details['recall']:.5f}, F1-Score: {test_details['f1']:.5f}"
    )
    return test_details


def pipeline(config: dict[str, str|float|dict[str, int]]) -> dict[str, Any]:
    """
    The pipeline works in 3 steps:
        - Performing configuration checks
        - Training and saving the model
        - Evaluating the model

    Trains the required model using the given configurations. To train any model for this
    assignment, pass the configurations to this function. The pipeline will train, save,
    and evaluate the model.
    :param config: A dictionary containing the configurations for training a model.
    The dictionary must define the following keys:
        - model: The model to train. Must be 'RNN', 'LSTM', 'GRU', or 'BiLSTM-CRF'.
        - dataset: The dataset to use. Must be 'NER' or 'ATE'.
        - embedding: The word-embedding to use. Must be 'Word2Vec', 'GloVe', or 'FastText'.
        - batch_size: The batch size for the dataloaders.
        - epochs: The number of epochs to train the model.
        - lr: The learning rate for the optimizer.
        - criterion: The loss function to use. Must be 'NLLLoss', 'CrossEntropyLoss', or 'CRF'.
        - optimizer: The optimizer to use. Must be 'Adam', 'Adagrad', or 'SGD'.
        - hyperparams: A dictionary containing the hyperparameters for the model. This
            must match the hyperparams expected by the model. The output_size and
            embedding_matrix will be added to this dictionary automatically.
        - device: The device to use for training. Default is 'cpu'.
    Optionally, the dictionary may also specify:
        - early_stopping_patience: The number of epochs to wait before stopping the training
            early (based on validation loss).
        - verbose: Whether to print the training progress or not. Default is False.
    Note: All values are case-insensitive.

    Returns a dictionary containing the trained model, the train, test, and validation dataloaders,
    and a dictionary of evaluation metrics for the test set.
    """

    check_config(config)

    train_set, test_set, val_set = load_dataset(config["dataset"], config["embedding"], config["verbose"])
    embedding_matrix = get_embedding_matrix(train_set.embedding)

    hyperparams = config.get("hyperparams", {})
    hyperparams["embedding_matrix"] = embedding_matrix
    hyperparams["output_size"] = len(train_set.encoder.classes_)
    model_name = config["model"]
    model = {"rnn": RNN, "lstm": LSTM, "gru": GRU, "bilstm-crf": BiLSTM_CRF}[model_name](**hyperparams)

    batch_size = config["batch_size"]
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

    criterion = config["criterion"]
    criterion = {"nllloss": torch.nn.NLLLoss, "crossentropyloss": torch.nn.CrossEntropyLoss, "crf": None}[criterion]()

    optimizer = config["optimizer"]
    optimizer = {"adam": torch.optim.Adam, "adagrad": torch.optim.Adagrad, "sgd": torch.optim.SGD}[optimizer]
    optimizer = optimizer(model.parameters(), lr=config["lr"])

    task = "t1" if config["dataset"] == "NER" else "t2"
    run = f"{task}_{model_name}_{config['embedding']}"

    model.to(config["device"])
    train(
        model, train_loader, val_loader, optimizer, criterion,
        config["epochs"], config["early_stopping_patience"], config["verbose"]
    )
    test_details = evaluate(model, test_loader, criterion)
    model_path = fr"Models/{run}.pt"
    torch.save(model.state_dict(), model_path)
    with open(fr"Models/{run}.pkl", "wb") as file:
        pickle.dump(test_details, file)

    return {
        "model": model, "encoder": train_set.encoder, **test_details,
        "train_loader": train_loader, "test_loader": test_loader, "val_loader": val_loader,
    }


def plot_learning_curve(model: torch.nn.Module) -> None:
    """
    Plots the loss and F1-score curves for the given model. The model must
    have been trained and must have a LOSSES and F1_SCORES attribute.
    Ideally, model should be RNN|LSTM|GRU|BiLSTM_CRF.
    """

    sns.set_theme(style="darkgrid")

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(model.LOSSES[:, 0], label="Train Loss")
    ax[0].plot(model.LOSSES[:, 1], label="Validation Loss")
    ax[0].set_title("Loss Curve")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(model.F1_SCORES[:, 0], label="Train F1-Score")
    ax[1].plot(model.F1_SCORES[:, 1], label="Validation F1-Score")
    ax[1].set_title("F1-Score Curve")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Macro F1-Score")
    ax[1].grid(True)
    ax[1].legend()

    plt.show()


def plot_confusion_matrix(confusion_matrix, labels: list[str]|None = None, size: tuple[int] = (7, 6)) -> None:
    """
    Plots the given confusion matrix. It is expected that the confusion
    matrix is normalized. Optionally, the labels for the classes can be
    passed to the function.
    """

    sns.set_theme(style="darkgrid")
    plt.figure(figsize=size)
    sns.heatmap(confusion_matrix, vmin=0.0, vmax=1.0)
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    if labels is not None:
        N = len(labels)
        plt.xticks(ticks=torch.arange(N) + 0.55, labels=labels, fontsize=8)
        plt.yticks(ticks=torch.arange(N) + 0.55, labels=labels, fontsize=8)
    plt.show()


if __name__ == "__main__":
    # An example of how to create and train a model
    CONFIG = dict(
        model="RNN",
        dataset="ATE",
        embedding="Word2Vec",
        batch_size=128,
        epochs=30,
        lr=1e-2,
        criterion="NLLLoss",
        optimizer="Adam",
        hyperparams=dict(
            input_size=300,
            hidden_size=128,
            num_layers=2
        ),
        early_stopping_patience=1,
        device="cpu",
        verbose=True
    )
    run = pipeline(CONFIG)
    # KEYS "model", "encoder", "train_loader", "test_loader", "val_loader", "accuracy",
    # "precision", "recall", "f1", "cf", and "loss" will now be available in 'run'.