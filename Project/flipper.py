import json
import pandas as pd
import torch

class Flipper:
    mapping: dict[str, str]

    def __init__(self, path: str):
        with open(path) as file:
            json_data = json.loads(file.read())
        self.mapping = dict()
        for word in json_data:
            try:
                if word["gender"] == "m":
                    m_word = word["word"]
                    f_word = word["gender_map"]["f"][0]["word"]
                    self.mapping[m_word] = f_word
                    self.mapping[f_word] = m_word
                elif word["gender"] == "f":
                    f_word = word["word"]
                    m_word = word["gender_map"]["m"][0]["word"]
                    self.mapping[m_word] = f_word
                    self.mapping[f_word] = m_word
            except KeyError:
                pass

    def __getitem__(self, key: str) -> str:
        return self.mapping.get(key, key)

    def is_gendered(self, sentence: str) -> bool:
        for word in sentence.split():
            if word in self.mapping:
                return True
        return False

    def flip(self, sentence: str) -> str:
        words = sentence.split()
        flipped_words = [self[word] for word in words]
        return " ".join(flipped_words)
    
    def flip_label(self,sentence: str):
        words = sentence.split()
        flipped_words = [self[word]==word for word in words]
        return flipped_words


    def flip_series(self, sentences: pd.Series) -> pd.Series:
        output = pd.Series(dtype="object")
        for i, sentence in sentences.items():
            output.at[i] = self.flip(sentence)
        return output

    def process_tokenizer(self, tokenizer):
        self.token_mapping = dict()
        for k,v in self.mapping.items():
            k_id = tokenizer.convert_tokens_to_ids(k)
            v_id = tokenizer.convert_tokens_to_ids(v)
            self.token_mapping[k_id] = v_id

    def process_tensor(self, tensor):
        mask = torch.zeros_like(tensor)
        for i in range(len(tensor)):
            for j in range(len(tensor[i])):
                if tensor[i][j] in self.token_mapping:
                    mask[i][j] = 1
        return mask

if __name__ == "__main__":
    gm = Flipper("gendered_words/gendered_words.json")
    print(gm.map["her"])