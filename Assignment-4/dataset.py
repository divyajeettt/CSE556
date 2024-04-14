import json
import pandas as pd
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

class MELD(Dataset):
    def __init__(self,path,EFR=False):
        self.EFR = EFR
        data = json.loads(open(path).read())
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.utterances = []
        self.emotions = []
        self.triggers = []
        for point in tqdm(data):
            self.utterances.append(tokenizer(point['utterances'], padding=True, truncation=True, return_tensors='pt'))
            self.emotions.append(point['emotions'])
            self.triggers.append(point['triggers'])
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(np.concatenate(self.emotions))
        self.emotions = [self.label_encoder.transform(e) for e in self.emotions]
        self.num_classes = len(self.label_encoder.classes_)

    def __len__(self):
        return len(self.utterances)
    
    def __getitem__(self,idx):
        if self.EFR:
            return self.utterances[idx],self.triggers[idx]
        return self.utterances[idx],self.emotions[idx]