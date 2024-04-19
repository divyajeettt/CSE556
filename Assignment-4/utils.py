import json
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from transformers import BertModel, BertTokenizer


class MELD(torch.utils.data.Dataset):
    def __init__(self, path, EFR=False):
        self.EFR = EFR
        data = json.loads(open(path).read())
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.utterances = []
        self.emotions = []
        self.triggers = []
        self.max_seq_len = 0
        for point in tqdm(data):
            self.utterances.append(tokenizer(point['utterances'], padding=True, truncation=True, return_tensors='pt'))
            self.emotions.append(point['emotions'])
            self.triggers.append(point['triggers'])
            self.max_seq_len = max(self.max_seq_len, len(point['emotions']))
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(np.concatenate(self.emotions))
        self.emotions = [self.label_encoder.transform(e) for e in self.emotions]
        self.num_classes = len(self.label_encoder.classes_)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self,idx):
        if self.EFR:
            return self.utterances[idx], np.array(self.triggers[idx], dtype=np.float32)
        return self.utterances[idx], self.emotions[idx]


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.2, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerBlock(torch.nn.Module):
    def __init__(self):
        super(TransformerBlock, self).__init__()
        self.key = torch.nn.Linear(768, 768)
        self.query = torch.nn.Linear(768, 768)
        self.value = torch.nn.Linear(768, 768)
        self.layer_norm = torch.nn.LayerNorm(768)
        self.attention = torch.nn.MultiheadAttention(768, 8)
        self.fc = torch.nn.Linear(768, 768)
        self.pos_emb = PositionalEncoding(768, max_len=100)

    def forward(self, x):
        #x : [seq_len,768]
        z = self.pos_emb(x)
        key = self.key(x)
        query = self.query(x)
        value = self.value(x)
        y = self.attention(query, key, value)[0]
        y = y + x
        y = self.layer_norm(y)
        z = self.fc(y)
        z = y + z
        z = self.layer_norm(z)
        return z


class ModelTransformer(torch.nn.Module):
    def __init__(self, task=1):
        super(ModelTransformer, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert.eval()
        self.seq_layer = TransformerBlock()
        self.fc = torch.nn.Linear(768, 128)
        self.fc2 = torch.nn.Linear(128, 7) if task == 1 else torch.nn.Linear(128, 2)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self,x):
        '''
        x : [num_utter, seq_len]
        '''
        with torch.no_grad():
            x = self.bert(**x).last_hidden_state
            x = x.mean(dim=1)
        if isinstance(self.seq_layer, TransformerBlock):
            x = self.seq_layer(x)
        else:
            x, _ = self.seq_layer(x)
        x = self.fc(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.softmax(x, dim=1)
        return x


class ModelGRU(ModelTransformer):
    def __init__(self, task=1):
        super(ModelGRU, self).__init__(task=task)
        self.seq_layer = torch.nn.GRU(768, 768)


def train(train_dataset, val_dataset, model, num_epochs=10, lr=1e-4, device='cuda', task=1):
    weight = None if task == 1 else torch.tensor([0.05, 1]).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)
    train_losses = []
    val_losses = []
    train_f1s = []
    val_f1s = []
    for i in range(num_epochs):
        model.train()
        true_ys = []
        pred_ys = []
        total_loss = 0
        for x, y in tqdm(train_dataset):
            x = {k: v.to(device) for k, v in x.items()}
            y = torch.from_numpy(y).to(device).type(torch.uint8)
            y = y if task == 1 else y.to(torch.uint8)
            if torch.sum(y.isnan()).item():
                continue
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.cpu().detach().item()
            true_ys.append(y.cpu().detach().numpy())
            pred_ys.append(y_pred.argmax(dim=1).cpu().detach().numpy())
        train_losses.append(total_loss/len(train_dataset))
        true_ys = np.concatenate(true_ys)
        pred_ys = np.concatenate(pred_ys)
        train_f1s.append(f1_score(true_ys, pred_ys, average='weighted'))
        val_loss = 0
        true_ys = []
        pred_ys = []
        print('Validating')
        with torch.no_grad():
            model.eval()
            for x, y in tqdm(val_dataset):
                x = {k: v.to(device) for k, v in x.items()}
                y = torch.from_numpy(y).to(device)
                y = y if task == 1 else y.to(torch.uint8)
                if torch.sum(y.isnan()).item():
                    continue
                y_pred = model(x)
                loss = loss_fn(y_pred, y).mean()
                val_loss += loss.cpu().detach().item()
                true_ys.append(y.cpu().detach().numpy())
                pred_ys.append(y_pred.argmax(dim=1).cpu().detach().numpy())
            val_losses.append(val_loss/len(val_dataset))
            true_ys = np.concatenate(true_ys)
            pred_ys = np.concatenate(pred_ys)
            val_f1s.append(f1_score(true_ys, pred_ys, average='weighted'))
        print(f'Epoch {i} Train Loss : {train_losses[-1]} Val Loss : {val_losses[-1]} Train F1 : {train_f1s[-1]} Val F1 : {val_f1s[-1]}')
        torch.save(model.state_dict(), 'model.pt')
    return train_losses, val_losses, train_f1s, val_f1s