from transformers import AutoTokenizer
import pandas as pd
import torch
from tqdm import tqdm

class Data(torch.utils.data.Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        df['training'] = df.apply(lambda x: str(x['sentence1']) + str(' [SEP] ') + str(x['sentence2']), axis=1)
        df['tokenized'] = df['training'].apply(lambda x: tokenizer.encode_plus(x, return_tensors='pt', padding=True, truncation=True,add_special_tokens=True))
        df['input_ids'] = df['tokenized'].apply(lambda x: x['input_ids'][0])
        df['attention_mask'] = df['tokenized'].apply(lambda x: x['attention_mask'][0])
        df['label'] = df['score'].apply(lambda x: torch.tensor(x/5.0))
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return self.df['input_ids'][idx], self.df['attention_mask'][idx], self.df['label'][idx]

def collate_fn(batch):
    input_ids = torch.nn.utils.rnn.pad_sequence([x[0] for x in batch],batch_first=True)
    attention_mask = torch.nn.utils.rnn.pad_sequence([x[1] for x in batch],batch_first=True)
    label = torch.stack([x[2] for x in batch]).unsqueeze(1)
    return input_ids, attention_mask, label

def train(model=None, train_dataloader=None,validation_dataloader=None,epochs=10,loss_fn=torch.nn.functional.mse_loss,
          lr=1e-3,batch_size=16):
    if train_dataloader is None:
        train_dataset = Data('Data/train.csv')
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    if validation_dataloader is None:
        validation_dataset = Data('Data/dev.csv')
        validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = 1e9
    for i in range(epochs):
        print('Training for epoch',i+1)
        total_loss = 0
        for input_ids, attention_mask, label in tqdm(train_dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(input_ids, attention_mask)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                total_loss += loss.item()
        print('Training loss',total_loss/len(train_dataloader))
        total_loss = 0
        print('Validation for epoch',i+1)
        with torch.no_grad():
            total_loss = 0
            for input_ids, attention_mask, label in tqdm(validation_dataloader):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                label = label.to(device)
                output = model(input_ids, attention_mask)
                loss = loss_fn(output, label)
                total_loss += loss.item()
            print('Validation loss',total_loss/len(validation_dataloader))
            if total_loss < best_val_loss:
                best_val_loss = total_loss
                torch.save(model.state_dict(), 'model.pth')
                print('Model saved')